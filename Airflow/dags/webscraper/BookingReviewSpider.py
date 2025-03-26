import scrapy
import json
import dateparser

class BookingReviewSpider(scrapy.Spider):
    name = "booking_reviews"

    def __init__(self, input_file, prev_month_start=None, prev_month_end=None, ignore_months=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_file = input_file
        self.ignore_months = ignore_months
        self.prev_month_start = prev_month_start
        self.prev_month_end = prev_month_end
    
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

            review_date = review.css("p.review_item_date::text").get().replace("Reviewed:", "").strip()
            parsed_date = dateparser.parse(review_date).date()

            if (self.ignore_months or self.prev_month_start <= parsed_date <= self.prev_month_end):

                tags = review.css("ul.review_item_info_tags li.review_info_tag ::text").getall()
                cleaned_tags = [tag.strip() for tag in tags if tag.strip() and tag.strip() != '•']

                yield {
                    "Hotel_Name": hotel_name,
                    "Hotel_Address": hotel_addr + hotel_country,
                    "Average_Score": hotel_score,
                    "Review_Date": review.css("p.review_item_date::text").get(),
                    "Reviewer_Score": review.css("span.review-score-badge::text").get(),
                    "Positive_Review": review.css("p.review_pos span[itemprop='reviewBody']::text").get(),
                    "Negative_Review": review.css("p.review_neg span[itemprop='reviewBody']::text").get(),
                    "Tags": cleaned_tags
                }

        next_page = response.css("a#review_next_page_link::attr(href)").get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)
