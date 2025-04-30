#!/bin/bash

ps --ppid `cat ./run.pid` | awk ' {print $1}' | awk 'NR!=1' >> ./run.pid
kill -TERM `cat ./run.pid`
echo > ./run.pid
echo shutdown
