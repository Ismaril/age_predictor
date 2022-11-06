import re
import time
import random
import os.path
import urllib3
import requests
import numpy as np
import pandas as pd
import constants as c

from PIL import Image
from bs4 import BeautifulSoup
from requests_tor import RequestsTor


def get_dataset(dataset_, finished_labels) -> list:
    """
    Return dataset that is going to be filtered based on data
    which were already downloaded.

    :param dataset_:
        Path to *.csv dataset containing classes.
    :param finished_labels:
        Path to *.csv containing info regarding last downloaded item.
    :return: list
    """

    # Check if some data were already scraped.
    try:
        # Find the last finished label.
        data_frame = pd.read_csv(finished_labels)
        last_item = np.array(data_frame)[-1, 0]

        # Filter source dataset by last finished label.
        source_data = pd.read_csv(dataset_)
        index_ = pd.Index(source_data["nconst"])
        index_ = index_.get_loc(last_item)
        return source_data[index_ + 1:]

    # Scraping starts for the first time.
    # (Meaning no data has been downloaded yet)
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
        Info: To use Tor requests, open manually Tor browser
        on your desktop and establish connection.

        :param user_data:
            Text data containing id of person and his age.
        :param user_agents:
            Insert user agents to appear as a browser.
        :param tor_request_html:
            True if request through tor else request only through 'Requests' lib.
        :param tor_request_img:
            True if request through tor else request only through 'Requests' lib.
        :param tor_timeout:
            Specify after what time should request time out.
        :param req_timeout:
            Specify after what time should request time out.
        :param nr_samples:
            Number of images to be downloaded.
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

        # Appear as browser making the request. (instead as "Requests")
        self.headers = {"User-Agent": random.choice(
            self.user_agents)}

        self.tor = RequestsTor(autochange_id=10,  # Change ip every x requests.
                               threads=8,
                               verbose=True)

    def get_webpage(self, person_id: str) -> BeautifulSoup | None:
        """
        Request a webpage either through Tor or Python Requests.

        :param person_id: ID of a person at IMDB.com

        :return: BeautifulSoup | None
        """
        try:
            web_page = f"https://www.imdb.com/name/{person_id}/?ref_=nv_sr_srsg_0"

            # Request webpage through tor.
            if self.tor_request_html:
                # print(f"\n{tor_request.get('http://httpbin.org/ip').text}")
                req = self.tor.get(web_page,
                                   timeout=self.tor_timeout,
                                   headers=self.headers)

            # Request webpage through python requests.
            else:
                req = requests.get(web_page,
                                   timeout=self.req_timeout,
                                   headers=self.headers)

            html_data = BeautifulSoup(req.text, "html.parser")
            return html_data

        # todo: Specify exact exception condition
        except:
            return None

    @staticmethod
    def parse_html(html_data: BeautifulSoup) -> str | None:
        """
        Parse html and get either None if there is no desired link to image
        or return the right link to image source. (*.png)

        :return: str | None
        """

        # If there is 'no-pic-image' found in html,
        #   means the desired image is not on the page.
        html_complete = html_data.find_all('div')
        pattern = re.compile('no-pic-image')
        matches = pattern.findall(str(html_complete))
        image = []
        for match in matches:
            image.append(match)
            if image:
                break
        if image:
            return None

        # Image link is present, lets find the image.
        else:
            html_complete = html_data.find_all('div', {'class': 'image'})
            pattern = re.compile('https.+jpg')
            matches = pattern.findall(str(html_complete))
            image = []
            for match in matches:
                image.append(match)
            return image[0]

    def get_image(self, img_source: str) -> urllib3.response.HTTPResponse | None:
        """
        Request a image either through Tor or Python Requests.
        urllib3.response.HTTPResponse object is returned both by Tor and Requests.

        :return: urllib3.response.HTTPResponse | None
        """

        # If not Stream=True, we would not be able to get the image from the
        # web page link.
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
    def save_img(raw_img: urllib3.response.HTTPResponse,
                 person_id: str,
                 person_age: str):
        """
        Save image from the web page.

        :param raw_img: Path to image
        :param person_id: Person's ID.
        :param person_age: Person's actual age.

        :return: None
        """
        with Image.open(raw_img) as img:
            img.save(os.path.join(c.IMAGES_RAW_DIR, f"{person_id}_{person_age}.png"))

    @staticmethod
    def save_last_iteration(person_id: str, person_age: str):
        """
        Save the id of last iterated item, to know where to continue next time,
        once the program runs again
        """
        with open(os.path.join(c.SCRAPING_DIR, "last_iteration.csv"), "w") as file:
            file.write(f"id, age\n{person_id}, {person_age}\n")

    @staticmethod
    def iteration_info(iteration_nr: int, time_start: float, time_stop: float):
        print(f"Iteration: {iteration_nr}")
        print(f"Elapsed: {time_stop - time_start}s")
        print(c.SEPARATOR)

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
