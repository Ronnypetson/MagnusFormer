if __name__ == '__main__':
  import re
  from glob import glob

  fns = glob('extracted/*.pgn')
  game_end = ['1-0', '0-1', '1/2-1/2']
  with open('complete_vocab.txt', 'r') as f:
    vocab = f.read().split('\n')
  vocab = set(vocab)
  suffixes = ['+', '#',
              '=N', '=B', '=R', '=Q',
              '=N+', '=B+', '=R+', '=Q+',
              '=N#', '=B#', '=R#', '=Q#']
  
  for fn in fns:
    simplified = []
    try:
      with open(fn, mode='r', encoding='utf-8') as f:
        data = f.read()
    except UnicodeDecodeError as e:
      try:
        with open(fn, mode='r', encoding='Windows-1252') as f:
          data = f.read()
      except UnicodeDecodeError:
        with open(fn, mode='r', encoding='Latin-1') as f:
          data = f.read()
    tokens = re.split('\.| |\n', data)
    for token in tokens:
      if token in vocab:
        simplified.append(token)
        if token in game_end:
          simplified.append('\n')
      else:
        for suffix in suffixes:
          if suffix in token and token[:-len(suffix)] in vocab:
            simplified.append(token)
            continue
    clean_fn = fn.split('/')[-1]
    simplified_fn = f'cleaned/{clean_fn}.clean'
    with open(simplified_fn, 'w') as f:
        f.write(' '.join(simplified))

