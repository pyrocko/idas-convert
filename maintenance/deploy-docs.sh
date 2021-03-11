#!/bin/bash
set -e
VERSION=v`python3 -c "import pkg_resources; print(pkg_resources.get_distribution('idas_convert').version)"`

if [ ! -f maintenance/deploy-docs.sh ] ; then
    echo "must be run from idas_convert's toplevel directory"
    exit 1
fi

cd doc
rm -rf build/$VERSION
make clean; make html $1
cp -r build/html build/$VERSION

read -r -p "Are your sure to update live docs at https://pyrocko.org/idas_convert/docs/$VERSION [y/N]?" resp
case $resp in
    [yY][eE][sS]|[yY] )
        rsync -av build/$VERSION pyrocko@hive:/var/www/pyrocko.org/idas_convert/docs;
        ;;
    * ) ;;
esac

read -r -p "Do you want to link 'current' to the just uploaded version $VERSION [y/N]?" resp
case $resp in
    [yY][eE][sS]|[yY] )
        echo "Linking idas_convert/docs/$VERSION to docs/current";
        ssh pyrocko@hive "rm -rf /var/www/pyrocko.org/idas_convert/docs/current; ln -s /var/www/pyrocko.org/idas_convert/docs/$VERSION /var/www/pyrocko.org/idas_convert/docs/current";
        ;;
    * ) ;;
esac

cd ..
