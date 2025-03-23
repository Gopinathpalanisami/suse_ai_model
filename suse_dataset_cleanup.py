from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import csv
import time

base_url = "https://www.suse.com/security/cve/"

def load_website(url):
    options = Options()
    options.add_argument("--headless=new")  # Run in headless mode (no GUI)
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    service = Service()
    driver = webdriver.Chrome(service=service, options=options)

    #try:
    #   driver.get(url)
    #    print("Page Title:", driver.title)
    #finally:
    #    driver.quit()

    try:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        cve_data = []

        for link in soup.select('a[href^="CVE-"]'):
            cve_id = link.text.strip()
            cve_link = link["href"].strip()

            #if cve_id.startswith("CVE-20"):
            if cve_id.startswith("CVE-20") and int(cve_id.split("-")[1]) >= 2014:
                description = get_cve_description(driver, base_url + cve_link)
                cve_data.append([cve_id, description])
                print(f"✅ {cve_id} - Description extracted")
				
		  with open("suse.csv", mode="w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["CVE ID", "Description"])
            csv_writer.writerows(cve_data)

        print(f"✅ CVE IDs successfully saved to suse.csv! Total CVEs: {len(cve_data)}")


    finally:
        driver.quit()

def get_cve_description(driver, cve_url):
    """Fetch description from the CVE detail page"""
    try:
        driver.get(cve_url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract description under <h4>Description</h4>
        description_tag = soup.find('h4', string="Description")
        if description_tag:
            #next_element = description_tag.find_next()
            next_element = description_tag.next_sibling
            if next_element and isinstance(next_element, str):
                description = next_element.strip()
            else:
                description = "No description available"
            #description = next_element.get_text(strip=True) if next_element else "No description available"
            #description = description_tag.find_next(string=True).strip()
        else:
            description = "No description available"

        return description

    except Exception as e:
        print(f"❌ Failed to fetch description for {cve_url}: {e}")
        return "Error fetching description"


if __name__ == "__main__":
    load_website("https://www.suse.com/security/cve/index.html")
	
	
	
