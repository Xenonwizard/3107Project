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

        hotel_name = response.css("a.standalone_header_hotel_link::text").get()
        hotel_addr = response.css("p.hotel_address::text").get()
        hotel_country = response.css("a.hotel_address_country::text").get()
        hotel_score = response.css("span.review-score-badge::text").get()

        for review in response.css("li.review_item"):
            yield {
                "Hotel_Name": hotel_name,
                "Hotel_Address": hotel_addr + hotel_country,
                "Average_Score": hotel_score,
                "Review_Date": review.css("p.review_item_date::text").get(),
                "Reviewer_Score": review.css("span.review-score-badge::text").get(),
                "Positive_Review": review.css("p.review_pos span[itemprop='reviewBody']::text").get(),
                "Negative_Review": review.css("p.review_neg span[itemprop='reviewBody']::text").get(),
            }

        next_page = response.css("a#review_next_page_link::attr(href)").get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)
