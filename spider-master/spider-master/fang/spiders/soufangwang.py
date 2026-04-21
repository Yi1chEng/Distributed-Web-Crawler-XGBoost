# -*- coding: utf-8 -*-
import scrapy
from fang.items import FangItem,OFangItem
import re
from scrapy_redis.spiders import RedisSpider
class SoufangwangSpider(RedisSpider):
    name = 'soufangwang'
    allowed_domains = ['fang.com']
    # start_urls = ['https://www.fang.com/SoufunFamily.htm']##开始爬取位置的url
    redis_key = 'fang:start_urls'

    def parse(self, response):
        # 提取省份信息以及发送省份的url
        province = None
        trs = response.xpath("//div[@id='c02']//tr")
        for tr in trs:
            # 省份信息
            tds = tr.xpath("./td[not(@class)]")
            province_thing = tds[0].xpath(".//text()").get().strip()
            if province_thing:
                province = province_thing
            # 不爬取海外的城市的房源
            if province == '其它':
                continue
            # 城市url和城市名
            city_links = tds[1].xpath("./a")
            for city_link in city_links:
                city = city_link.xpath("./text()").get()
                city_url = city_link.xpath("./@href").get()
                if 'bj' not in city_url:
                    new_url = 'newhouse.fang'.join(city_url.split('fang')) + 'house/s/'
                    esf_url = 'esf.fang'.join(city_url.split('fang'))
                else:
                    new_url = 'https://newhouse.fang.com/house/s/'
                    esf_url = 'https://esf.fang.com/'
                yield response.follow(url=new_url, callback=self.parse_new, meta={"info": (province, city)})
                yield response.follow(url=esf_url, callback=self.parse_esf, meta={"info": (province, city)})
                
    def parse_new(self,response):
        lis = response.xpath("//div[@class='nhouse_list']/div/ul/li")
        province, city = response.meta['province'],response.meta['city']
        for li in lis:
            name = re.sub(r'\s','',''.join(li.xpath(".//div[contains(@class,'house_value')]//a/text()").getall()))
            rank = 4.5
            house_info = re.sub(r'\s','',''.join(li.xpath(".//div[contains(@class,'house_type')]//text()").getall()))
            position = li.xpath(".//div[@class='address']/a/@title").get()
            origin_url = response.urljoin(li.xpath(".//div[@class='address']/a/@href").get())
            onsale = li.xpath(".//div[@class='fangyuan']/span/text()").get()
            special = ' '.join(li.xpath(".//div[@class='fangyuan']/a/text()").getall())
            price = re.sub(r'\s','',''.join(li.xpath(".//div[@class='nhouse_price']//text()").getall()))
            item = FangItem(name=name,rank=rank,house_info=house_info,position=position,onsale=onsale,
            special=special,price=price,origin_url=origin_url,city=city,province=province)
            yield item
        next_ = response.xpath("//div[@class='page']//a[@class='next']/@href").get()
        if next_:
            next_url = response.urljoin(next_)
            yield response.follow(url=next_url,callback=self.parse_new,meta={'province':province,
                        'city':city})
    def parse_esf(self,response):
        dls = response.xpath("//div[contains(@class,'shop_list')]//dl[@data-bg]")
        province,city = response.meta['province'],response.meta['city']
        for dl in dls:
            name = re.sub(r'\s','',dl.xpath(".//span[@class='tit_shop']/text()").get())
            house_info = re.sub(r'\s','',''.join(dl.xpath(".//p[@class='tel_shop']//text()").getall()))
            position = dl.xpath(".//p[@class='add_shop']/span/text()").get()
            subway = dl.xpath(".//p[contains(@class,'clearfix')]/span[not(@class='\\')]/text()").get()
            price = re.sub(r'\s','',''.join(dl.xpath(".//dd[@class='price_right']/span[@class='red']//text()").getall()))
            avg_price = dl.xpath(".//dd[@class='price_right']/span[2]//text()").get()
            item = OFangItem(province=province,city=city,name=name,house_info=house_info,position=position,subway=subway,
                            price = price,avg_price=avg_price
                            )
            yield item
        next_ = response.xpath("//div[@class='page_al']/p[1]/@href").get()
        if next_:
            next_url = response.urljoin(next_)
            yield response.follow(url=next_url,callback=self.parse_esf,meta={'province':province,
                        'city':city
                        })
