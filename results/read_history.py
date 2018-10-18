#!/usr/bin/env python

import pickle


project_name = 'wine'
fold_id = ''

history = pickle.load(open('./' + project_name + '/' + fold_id + 'run_history.pk','rb'))

import pdb; pdb.set_trace()
