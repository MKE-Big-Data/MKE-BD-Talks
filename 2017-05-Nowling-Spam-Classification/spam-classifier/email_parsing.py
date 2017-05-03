"""
Copyright 2016 Ronald J. Nowling

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from email.parser import Parser
import os

from bs4 import BeautifulSoup

def _parse_message(message):
    body = ""

    if message.is_multipart():
        for part in message.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))

            # skip any attachments
            if ctype == 'text/html' and 'attachment' not in cdispo:
                body = part.get_payload(decode=True)
                break
            elif ctype == 'text/txt' and 'attachment' not in cdispo:
                body = part.get_payload(decode=True)
                break
    # not multipart - i.e. plain text, no attachments, keeping fingers crossed
    else:
        body = message.get_payload(decode=True)
            
    return message["To"], message["From"], BeautifulSoup(body, 'html.parser').get_text()

def stream_email(data_dir):
    email_parser = Parser()
    
    index_flname = os.path.join(data_dir, "full", "index")
    with open(index_flname) as index_fl:
        for idx, ln in enumerate(index_fl):
            category, email_fl_suffix = ln.strip().split()
            if category == "ham":
                label = 0
            elif category == "spam":
                label = 1
            
            # strip ../ prefix from path
            email_flname = os.path.join(data_dir, email_fl_suffix[3:])
            with open(email_flname) as email_fl:
                message = email_parser.parse(email_fl)
                to, from_, body = _parse_message(message)

                yield (label, to, from_, body)

def read_all_emails(data_dir):
    stream = stream_email(data_dir)
    bodies = []
    labels = []
    for idx, (label, to, from_, body) in enumerate(stream):
        bodies.append(body)
        labels.append(label)

    return bodies, labels


