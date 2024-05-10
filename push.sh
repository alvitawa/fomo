#!/bin/bash
rsync --links -r --exclude-from=.rsyncignore -e ssh --delete . ataboadawarmer@snellius.surf.nl:workspace
