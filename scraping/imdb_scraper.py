import re
import time
import random
import requests
import numpy as np
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup
from requests_tor import RequestsTor

SEPARATOR = "-" * 100
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
    "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1",
]


def get_dataset(dataset_, finished_labels):
    # check if some data were already scraped
    try:
        # find the last finished label
        data_frame = pd.read_csv(finished_labels)
        last_item = np.array(data_frame)[-1, 0]

        # filter source dataset by last finished label
        source_data = pd.read_csv(dataset_)
        index_ = pd.Index(source_data["nconst"])
        index_ = index_.get_loc(last_item)
        return source_data[index_ + 1:]

    # scraping starts for the first time
    except pd.errors.EmptyDataError:
        source_data = pd.read_csv(dataset_)
        return source_data


class IMDBScraper:
    def __init__(self, user_data: list,
                 user_agents: list,
                 tor_request_html=True,
                 tor_request_img=True,
                 tor_timeout=30,
                 req_timeout=15,
                 nr_samples=None):
        """
        :param user_data: text data containing id of person and his age
        :param user_agents: insert user agents to appear as a browser
        :param tor_request_html: True if request through tor else request only through 'Requests' lib
        :param tor_request_img: True if request through tor else request only through 'Requests' lib
        :param tor_timeout: specify after what time should request time out
        :param req_timeout: specify after what time should request time out
        :param nr_samples: number of images to be downloaded
        """
        self.user_data = user_data
        self.user_agents = user_agents
        self.person_id = np.array(self.user_data)[:, 1]
        self.person_age = np.array(self.user_data)[:, -1]
        self.tor_request_html = tor_request_html
        self.tor_request_img = tor_request_img
        self.tor_timeout = tor_timeout
        self.req_timeout = req_timeout
        self.nr_samples = nr_samples
        self.headers = {"User-Agent": random.choice(
            self.user_agents)}  # appear as browser making the request (instead as "Requests")
        self.tor = RequestsTor(autochange_id=10,  # change ip every x requests
                               threads=8,
                               verbose=True)

    def get_webpage(self, person_id):
        try:
            # request webpage through tor
            if self.tor_request_html:
                # print(f"\n{tor_request.get('http://httpbin.org/ip').text}")
                req = self.tor.get(f"https://www.imdb.com/name/{person_id}/?ref_=nv_sr_srsg_0",
                                   timeout=self.tor_timeout,
                                   headers=self.headers)

            # request webpage through python requests
            else:
                req = requests.get(f"https://www.imdb.com/name/{person_id}/?ref_=nv_sr_srsg_0",
                                   timeout=self.req_timeout,
                                   headers=self.headers)

            html_data = BeautifulSoup(req.text, "html.parser")
            return html_data
        except:
            return None

    @staticmethod
    def parse_html(html_data):
        """
        Parse html and get either None if there is no desired link to image
        or return the right link to image source (.png)
        """
        # if there is 'no-pic-image' found in html, means the desired image is not
        # on the page
        html_complete = html_data.find_all('div')
        pattern = re.compile('no-pic-image')
        matches = pattern.findall(str(html_complete))
        image = []
        for match in matches:
            image.append(match)
            if image: break
        if image:
            return None

        # image link is present, lets find it:
        else:
            html_complete = html_data.find_all('div', {'class': 'image'})
            pattern = re.compile('https.+jpg')
            matches = pattern.findall(str(html_complete))
            image = []
            for match in matches:
                image.append(match)
            return image[0]

    def get_image(self, img_source):
        """Get image either through tor or normal python requests"""
        try:
            if self.tor_request_img:
                req = self.tor.get(img_source,
                                   stream=True,
                                   timeout=self.tor_timeout,
                                   headers=self.headers).raw
            else:
                req = requests.get(img_source,
                                   stream=True,
                                   timeout=self.req_timeout,
                                   headers=self.headers).raw
            return req
        except:
            return None

    @staticmethod
    def save_img(raw_img, person_id, person_age):
        with Image.open(raw_img) as img:
            img.save(f"C:/Users/lazni/PycharmProjects/Age_Predictor/images/{person_id}_{person_age}.png")

    @staticmethod
    def save_last_iteration(person_id, person_age):
        """
        Save the id of last iterated item, to know where to continue next time,
        once the program runs again
        """
        with open("last_iteration.csv", "w") as file:
            file.write(f"id, age\n{person_id}, {person_age}\n")

    @staticmethod
    def iteration_info(iteration_nr, time_start, time_stop):
        print(f"Iteration: {iteration_nr}")
        print(f"Elapsed: {time_stop - time_start}s")
        print(SEPARATOR)

    def main(self):
        for i, (person_id, person_age) in enumerate(zip(self.person_id, self.person_age)):
            self.save_last_iteration(person_id, person_age)
            measure_time_start = time.perf_counter()

            html = self.get_webpage(person_id)
            if html is None:
                print(f"\nRequest timed out (getting html)")
                self.iteration_info(i, measure_time_start, time.perf_counter())
                continue
            else:
                link_to_img = self.parse_html(html)
                # print(f"\n{link_to_img=}")

            if link_to_img is None:
                print("Page has no desired image")
                self.iteration_info(i, measure_time_start, time.perf_counter())
                continue

            raw_img = self.get_image(link_to_img)
            if raw_img is None:
                print(f"\nRequest timed out (getting image)")
                self.iteration_info(i, measure_time_start, time.perf_counter())
                continue
            else:
                self.save_img(raw_img, person_id, person_age)
                print(f"\n{person_id=}, {person_age=}")
                self.iteration_info(i, measure_time_start, time.perf_counter())

            if self.nr_samples is None:
                pass
            elif i == self.nr_samples - 1:
                print(f"Scraping finished due to set limit")
                break

        if self.nr_samples is None:
            # clear all contents in file that keeps last iteration
            with open("last_iteration.csv", "w") as file:
                file.write("")
            print(
                f"Scraping finished dataset with {len(self.person_id) if self.nr_samples is None else self.nr_samples} iterations")


##########################################################################################################
dataset = get_dataset("C:/Users/lazni/PycharmProjects/Age_Predictor/source_data/source_data_0.csv",
                      "last_iteration.csv")
scraper = IMDBScraper(user_data=dataset,
                      user_agents=USER_AGENTS,
                      tor_request_html=False,
                      tor_request_img=False,
                      tor_timeout=30,
                      req_timeout=15,
                      nr_samples=None)
scraper.main()
