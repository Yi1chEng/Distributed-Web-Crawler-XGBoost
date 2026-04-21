# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class FangItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    name = scrapy.Field()
    rank = scrapy.Field()
    house_info = scrapy.Field()
    position = scrapy.Field()
    onsale = scrapy.Field()
    special = scrapy.Field()
    price = scrapy.Field()
    origin_url = scrapy.Field()
    province = scrapy.Field()
    city = scrapy.Field()
    

class OFangItem(scrapy.Item):
    province = scrapy.Field()
    city = scrapy.Field()
    name = scrapy.Field()
    house_info = scrapy.Field()
    position = scrapy.Field()
    subway =scrapy.Field()
    price = scrapy.Field()
    avg_price = scrapy.Field()




