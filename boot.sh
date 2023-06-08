#!/bin/bash
exec gunicorn -b :5555 --access-logfile - --error-logfile - myapp:app