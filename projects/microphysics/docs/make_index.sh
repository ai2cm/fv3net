#!/bin/bash

cat << EOF > index.html
<h1>Microphysics Emulation Reports Index</h1>
Run date: $(date) <br />
Git revision: $(git rev-parse HEAD) <br />
Host: $HOSTNAME <br />
OS: $(uname) <br />
EOF

echo "<ul>" >> index.html
for f in $@; do 
echo "<li><a href=\"$f\">$f</a></li>"  >> index.html
done
echo "</ul>" >> index.html