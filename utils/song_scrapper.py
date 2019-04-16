import urllib3
from bs4 import BeautifulSoup
import pandas as pd 
import time
import pickle
#song webscraper 
#surpress warnings for now..
urllib3.disable_warnings()
http = urllib3.PoolManager()
#defaults to beatles currently
def scrape_band(SONG_URL='https://www.azlyrics.com/b/beatles.html', BAND_NAME='beatles'):
        song = http.request('GET', SONG_URL)
        soup = BeautifulSoup(song.data, 'html.parser')
        links = soup.findAll('a')
        clean_links = []
        for link in links:
                url = link.get('href')
                if BAND_NAME in url:
                        name = str(link.text).lower()
                        url = 'https://www.azlyrics.com' + url[2:]
                        clean_links.append([name, url])
        print(clean_links)
        print('started collecting...')
        df = pd.DataFrame(columns=['Name', 'Lyrics'])
        #create df of songs:
        count = 1
        for link in clean_links:
                try:
                        name = str(link[0]).lower()
                        print('name: ', name)
                        #url = 'https://www.azlyrics.com' + url[2:]
                        url = str(link[1])
                        print('url: ', url)
                        curr_song = http.request('GET', url)
                        soup = BeautifulSoup(curr_song.data, 'html.parser')
                        song = soup.find_all('div', class_=False)[::-1]
                        song = song[0].text.lower()
                        temp = {'Name': name, 'Lyrics': song}
                        df = df.append(temp, ignore_index=True)
                        if count % 10 == 0 and count != 1:
                                print('writing to file: ', count)
                                df.to_pickle('beatles.pkl', compression='gzip')
                        print('sleeping, curr count: ', count)
                        time.sleep(1.5)
                except Exception as e:
                        print('something went wrong: ', e)
                        continue
                count += 1
        
