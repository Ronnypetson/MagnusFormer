if __name__ == '__main__':
  import os
  
  url_base = 'https://theweekinchess.com/zips'
  for idx in range(920, 1431):
    fname = f'twic{idx}g.zip'
    print('Downloading', fname)
    url = f'{url_base}/{fname}'
    os.system(f'wget {url}')
    os.system(f'mv {fname} downloaded/{fname}')

