#!/bin/sh
python classifier.py>output
python f_selection.py
python wiki_extract.py
python co_occurrence.py
python m_classifier.py>>output
