from urllib.parse import urlparse
import socket
import ipaddress
import requests
from typing import Set
import re

from domain.exceptions import SecurityError, UnsupportedDomainToScrape


EXTRA_BLOCKLIST = [
    # Link-local
    ipaddress.ip_network("169.254.0.0/16"),
    # Carrier-Grade NAT (CGNAT)
    ipaddress.ip_network("100.64.0.0/10"),
    # Benchmarking / Testing
    ipaddress.ip_network("198.18.0.0/15"),
    # Documentation ranges (should never be routable)
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    # Reserved "future use"
    ipaddress.ip_network("240.0.0.0/4"),
    # Multicast
    ipaddress.ip_network("224.0.0.0/4"),
    # IPv6 link-local
    ipaddress.ip_network("fe80::/10"),
    # IPv6 unique local
    ipaddress.ip_network("fc00::/7"),
    # IPv6 documentation
    ipaddress.ip_network("2001:db8::/32"),
]

ALLOWED_DOMAINS = {
    "www.google.com": {"require_path": lambda p: p.startswith("/maps")},
    "maps.google.com": {"require_path": lambda p: p.startswith(("/maps", "/search"))},
    "maps.app.goo.gl": {"require_path": lambda p: True},
}


def _address_is_safe(ip_str: str) -> bool:
    addr = ipaddress.ip_address(ip_str)
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_unspecified
    ):
        return False
    for net in EXTRA_BLOCKLIST:
        if addr in net:
            return False
    return True


def _resolve_all_ips(hostname: str) -> Set[str]:
    results = set()
    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise SecurityError(
            possible_attack="dns_resolution_failed", received_hostname=hostname
        ) from e

    for info in infos:
        sockaddr = info[4]
        ip = sockaddr[0]
        results.add(ip)
    return results


def _expand_url(url: str, max_redirect: int = 2):
    session = requests.Session()
    session.max_redirects = max_redirect

    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        if response.status_code in (405, 501):
            response = requests.get(url, allow_redirects=True, timeout=10)
        return response.url
    except requests.exceptions.TooManyRedirects:
        raise SecurityError(possible_attack="too_many_redirect", received_url=url)
    except requests.exceptions.RequestException:
        raise


def validate_gmaps_url(url: str):
    try:
        if _match := re.search(r"maps.app.goo.gl", url):
            parsed = urlparse(_expand_url(url))
            hostname = parsed.hostname
        else:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
        if parsed.scheme not in {"https", "http"}:
            raise SecurityError(
                received_scheme=parsed.scheme,
                supported_scheme="https, http",
                attack_vector="scrape_url_input",
                possible_attack="unsupported_scheme",
            )
        if hostname not in ALLOWED_DOMAINS:
            raise UnsupportedDomainToScrape(platform="google maps")
        path_ok = ALLOWED_DOMAINS[hostname]["require_path"](parsed.path)
        if not path_ok:
            raise SecurityError(
                possible_attack="invalid_path_for_hostname",
                received_path=parsed.path,
                received_hostname=hostname,
            )
        ips = _resolve_all_ips(hostname)
        if not ips:
            raise SecurityError(
                possible_attack="no_dns_records", received_hostname=hostname
            )
        for ip in ips:
            if not _address_is_safe(ip):
                raise SecurityError(
                    possible_attack="ssrf_blocked",
                    received_hostname=hostname,
                    detected_ip=ip,
                )
        return parsed.geturl()
    except SecurityError:
        raise
    except Exception as e:
        raise SecurityError(possible_attack="validation_error") from e
