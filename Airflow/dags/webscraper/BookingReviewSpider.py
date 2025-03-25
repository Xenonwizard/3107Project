import scrapy
import json

class BookingReviewSpider(scrapy.Spider):
    name = "booking_reviews"
    
    def start_requests(self):
        with open(self.input_file) as f:
            urls = json.load(f)
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        yield {
            'Hotel_Name': response.css('.standalone_header_hotel_link::text').get(),
            'Hotel_Address': response.css('.hotel_address::text').get(),
            'Hotel_Score' : response.css('.review-score-badge::text').get(),
        }
