import scrapy
import json

class BookingReviewSpider(scrapy.Spider):
    name = "booking_reviews"

    def __init__(self, input_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_file = input_file
    
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

            tags = review.css("ul.review_item_info_tags li.review_info_tag ::text").getall()
            cleaned_tags = [tag.strip() for tag in tags if tag.strip() and tag.strip() != '•']

            yield {
                "Hotel_Address": hotel_addr + hotel_country,
                "Review_Date": review.css("p.review_item_date::text").get().replace("Reviewed:", "").strip(),
                "Average_Score": hotel_score,
                "Hotel_Name": hotel_name,
                "Reviewer_Nationality": review.css("span.reviewer_country span[itemprop='name']::text").get(),
                "Negative_Review": review.css("p.review_neg span[itemprop='reviewBody']::text").get(),
                "Positive_Review": review.css("p.review_pos span[itemprop='reviewBody']::text").get(),
                "Reviewer_Score": review.css("span.review-score-badge::text").get(),
                "Tags": cleaned_tags
            }

        next_page = response.css("a#review_next_page_link::attr(href)").get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)
