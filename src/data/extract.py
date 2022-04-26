if __name__ == '__main__':
  import zipfile
  from glob import glob
  
  fns = glob('downloaded/*.zip')
  
  for fn in fns:
    with zipfile.ZipFile(fn, 'r') as zip_ref:
      zip_ref.extractall('extracted')
  
