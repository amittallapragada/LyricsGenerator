import urllib3
from bs4 import BeautifulSoup
import pandas as pd 
#surpress warnings for now..
urllib3.disable_warnings()

#Hopefully works for azlyrics
SONG_URL = 'https://www.azlyrics.com/g/greenday.html'
BAND_NAME = 'greenday'
http = urllib3.PoolManager()

song = http.request('GET', SONG_URL)
soup = BeautifulSoup(song.data, 'html.parser')

links = soup.findAll('a')
df = pd.DataFrame(columns=['Name', 'Lyrics'])
#create df of songs:
for link in links:
    url = link.get('href')
    if url and BAND_NAME in url:
        try:
            name = str(link.text).lower()
            url = 'https://www.azlyrics.com' + url[2:]
            curr_song = http.request('GET', url)
            soup = BeautifulSoup(curr_song.data, 'html.parser')
            song = soup.find_all('div', class_=False)[::-1]
            song = song[0].text.lower()
            temp = {'Name': name, 'Lyrics': song}
            df = df.append(temp, ignore_index=True)
        except Exception as e:
            print('something went wrong: ', e)
            continue
        
print(df.head())
print(len(df))
df.to_pickle('green_day_lyrics.pkl', compression='gzip')
print('DONE.')
