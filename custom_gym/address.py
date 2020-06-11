from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time

def find_address(driver, building_name, district='cbd'):
    if district == 'cbd':
        building_name += ' 종로'

    url = 'https://map.naver.com/'
    driver.get(url)

    search = driver.find_element_by_id('search-input')
    search.send_keys(building_name)
    btn = driver.find_element_by_xpath('//*[@id="header"]/div[1]/fieldset/button')
    btn.click()

    btns = driver.find_elements_by_class_name('roadname')
    addr = ''
    for btn in btns:
        btn.click()
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        try:
            addr = soup.select('dd.info_road')[0].text[5:]
            city, gu, dong, etc = addr.split(' ')
            print(city, gu, dong, etc)
            break
        except:
            continue
    if addr == '' : 
        print('주소를 찾지 못햇습니다...')
        return [False, None]


    #####부동산 정보 조회 시스템#####
    url = 'http://kras.seoul.go.kr/land_info/info/baseInfo/baseInfo.do'
    driver.get(url)
    count = 0
    while True:
        if count > 10:
            print('등기본등본 조회를 실패하였습니다.')
            return [False, None]
        count += 1
        try:
            city_button = driver.find_element_by_xpath('//*[@id="sidonm"]')
            city_button.click()
            city_button.send_keys(city)
            gu_btn = driver.find_element_by_xpath('//*[@id="sggnm"]')
            gu_btn.click()
            gu_btn.send_keys(gu)
            dong_btn = driver.find_element_by_xpath('//*[@id="umdnm"]')
            dong_btn.click()
            dong_btn.send_keys(dong)
            etc_text = driver.find_element_by_xpath('//*[@id="textfield"]')
            etc_text.clear()
            etc_text.send_keys(etc)
            btn = driver.find_element_by_xpath('//*[@id="searching"]/a')
            btn.click()
            break
        except:
            time.sleep(1)
            continue

    count = 0
    while True:
        if count > 10:
            print('등기본등본 조회를 실패하였습니다.')
            return [False, None]
        count += 1
        try:
            driver.find_element_by_xpath('//*[@id="tab0301"]/li[3]/a').click()
            driver.implicitly_wait(1)
            driver.find_element_by_xpath('//*[@id="bldInfo_print"]/table/tbody/tr').click()
            break
        except:
            time.sleep(1)
            continue

    count = 0
    while True:
        count += 1
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        labels = soup.select('td label')
        totarea = ''
        useaprDay = ''
        for label in labels:
            if label['for'] == 'totarea':
                totarea = label.text.strip()
            elif label['for'] == 'useaprDay':
                useaprDay = label.text.strip()

        if totarea != '': break
        else : time.sleep(1)

        if count > 10 :
            print('등기부등본 데이터를 조회할 수 없습니다.')
            return [False, None]

    tables = soup.select('#bldTitleInfo > table')
    levels = None
    parks = None
    elevators = None
    for table in tables:
        if table['summary'] == '층별현황':
            levels = table
        elif table['summary'] == '주차장':
            parks = table
        elif table['summary'] == '승강기':
            elevators = table
    levels = levels.select('tr')[1:]

    level_list = []
    under = 0
    over = 0
    for level in levels:
        temp = level.select('td')[1].text
        if temp in level_list : continue
        level_list.append(temp)
        if temp[0] == '지' : under += 1
        elif temp[0] != '옥' : over += 1

    num_parking = 0
    parks = parks.select('td')
    for park in parks:
        num_parking += int(park.text[:park.text.find('대')])

    num_elevators = 0
    elevators = elevators.select('td')
    for elevator in elevators:
        num_elevators += int(elevator.text[:elevator.text.find('대')])

    return [True, [totarea, useaprDay, under, over, num_parking, num_elevators]]

if __name__ == '__main__':
    options = webdriver.ChromeOptions()
    #options.add_argument('headless')
    #options.add_argument('window-size=1920x1080')
    #options.add_argument("disable-gpu")
    options.add_argument('--start-fullscreen')
    driver = webdriver.Chrome('./chromedriver', chrome_options=options)
    driver.implicitly_wait(3) # 암묵적으로 웹 자원을 (최대) 3초 기다리기

    names = [
    '교보생명빌딩'
    ,'D타워'
    ,'타워8'
    ,'그랑서울'
    ,'스탠다드차티드은행빌딩'
    ,'부영을지빌딩'
    ,'교원내외빌딩'
    ,'페럼타워'
    ,'미래에셋센터원빌딩'
    ,'종로타워']

    find_address(driver, names[1])
    driver.close()
