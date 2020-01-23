
# search file "doesn't include  `format`" in file name
find ./ -type f | grep -P '^(?!.*format).*$'
