if __name__ == '__main__':
  from glob import glob
  
  fns = glob('cleaned/*.clean')
  joint = ''
  
  for fn in fns[:100]:
    with open(fn, 'r') as f:
      joint += f.read()
  
  joint_fn = 'clean_joint_100.cpgn'
  with open(joint_fn, 'w') as f:
    f.write(joint)

