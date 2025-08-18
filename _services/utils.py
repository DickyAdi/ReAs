import ipaddress

def mask_ip(ip:str):
    try:
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.version == 4:
            blocks = ip.split('.')
            return '.'.join(blocks[:2] + ['xxx', 'xxx'])
        else:
            blocks = ip.split(':')
            return ':'.join(blocks[:2] + ['xxxx'] * (len(blocks) - 2))
    except ValueError:
        return 'Unkown'