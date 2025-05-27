import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class PropertyScraper:
    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    def extract_postcode_and_name(self, property_index):
        match = re.match(r"([A-Z]{1,2}\d{1,2} \d[A-Z]{2})\s+(.+)", property_index, re.IGNORECASE)
        if match:
            print("match group 1", match.group(1))
            print("match group 2", match.group(2))
            return match.group(1), match.group(2)
        print("not matched")
        return property_index, ""

    def scrape_property_details(self, postcode, property_name, retries=3, delay=5):
        url = f"https://propertychecker.co.uk/results/?postcode={postcode}"
  
        for attempt in range(retries):
            try:
                self.driver.get(url)
                
                # Wait for the page to load completely
                WebDriverWait(self.driver, 30).until(
                    EC.visibility_of_element_located((By.XPATH, "//a[contains(@href, '/property-details')]"))
                )
                
                property_links = self.driver.find_elements(By.XPATH, "//a[contains(@href, '/property-details')]")
                selected_url = None
                property_name_cleaned = re.sub(r"\s*-\s*", "-", property_name).replace(",", "").strip()


                for link in property_links:
                    name = link.text.strip()
                    name_cleaned = re.sub(r"\s*-\s*", "-", name).replace(",", "").strip()


                    if property_name_cleaned.lower() in name_cleaned.lower():
                        selected_url = link.get_attribute("href")
                        break  
                
                if not selected_url:
                    print(f"Property not found: {property_name}")
                    return None
                
                self.driver.get(selected_url)
                
                # Wait for the details page to load completely
                WebDriverWait(self.driver, 30).until(
                    EC.visibility_of_element_located((By.XPATH, "//dt[contains(text(),'Total Floor Area')]/following-sibling::dd"))
                )
                
                floor_area = self.driver.find_element(By.XPATH, "//dt[contains(text(),'Total Floor Area')]/following-sibling::dd").text
                habitable_rooms = self.driver.find_element(By.XPATH, "//dt[contains(text(),'Habitable Rooms')]/following-sibling::dd").text
                heated_rooms = self.driver.find_element(By.XPATH, "//dt[contains(text(),'Heated Rooms')]/following-sibling::dd").text
                tenure = self.driver.find_element(By.XPATH, "//dt[contains(text(),'Tenure')]/following-sibling::dd").text
                form = self.driver.find_element(By.XPATH, "//dt[contains(text(),'Form')]/following-sibling::dd").text
                type=self.driver.find_element(By.XPATH,"//dt[contains(text(),'Type')]/following-sibling::dd").text
                year_built=self.driver.find_element(By.XPATH,"//dt[contains(text(),'Year Built')]/following-sibling::dd").text
                new_build=self.driver.find_element(By.XPATH,"//dt[contains(text(),'New Build')]/following-sibling::dd").text
                current_energy_rating=self.driver.find_element(By.XPATH,"//dt[contains(text(),'Current Energy Rating')]/following-sibling::dd").text
                est_value=self.driver.find_element(By.XPATH,"//dt[contains(text(), 'Est. Value')]/following-sibling::dd").text

                if floor_area == "" or habitable_rooms == "" or heated_rooms == "":
                    print(f"Empty value found for {property_name}. Retrying...")
                    raise ValueError("Empty value found")
                
                return {
                    "Total Floor Area": floor_area,
                    "Habitable Rooms": habitable_rooms,
                    "Heated Rooms": heated_rooms,
                    "Tenure": tenure,
                    "Form": form,
                    "Type": type,
                    "Year Built":year_built,
                    "New Build":new_build,
                    "Current Energy Rating":current_energy_rating,
                    "Estimated Value":est_value
                }
            
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {property_name}. Error: {e}")
                if attempt < retries - 1:
                    # Refresh the page or reconnect
                    self.driver.refresh()
                    time.sleep(delay)
                else:
                    print(f"Failed to extract details for {property_name} after {retries} attempts")
                    return None
    
    
    def close(self):
        self.driver.quit()