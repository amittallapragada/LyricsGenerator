#testing song scrapper
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from utils import song_scrapper as scrape 

scrape.scrape_band(SONG_URL='https://www.azlyrics.com/b/beatles.html', BAND_NAME='beatles')