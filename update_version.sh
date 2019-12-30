echo "Updating version number in files: from $1 to $2"
sed -i "s/$1/$2/g" setup.py
sed -i "s/$1/$2/g" ./deep_learning_utils/__init__.py
sed -i "s/$1/$2/g" ./doc/source/conf.py
