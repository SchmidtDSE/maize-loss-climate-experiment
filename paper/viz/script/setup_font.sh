mkdir font_build
cd font_build
wget https://github.com/uswds/public-sans/releases/download/v2.001/public-sans-v2.001.zip
unzip public-sans-v2.001
cd ..
mkdir font
mv font_build/fonts/otf/PublicSans-Regular.otf font/PublicSans-Regular.otf
