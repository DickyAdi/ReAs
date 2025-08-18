from enum import Enum

class Tier(str, Enum):
    Base = 'base'
    Mid = 'mid'
    Pro = 'pro'

TIER_LEVEL = {
    "base" : 0,
    'mid' : 1,
    'pro' : 2
}