import torch
import argparse
from time import time
from lastDataset import dataset
from models.newmodel import model 
from pargs import pargs,dynArgs
#import utils.eval as evalMetrics
import glob
import logging
from eval import Evaluate

logger = logging.getLogger('myapp')
hdlr = logging.FileHandler('eval-graph.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)



evaluator = Evaluate()

def tgtreverse(tgts,entlist,order):
  entlist = entlist[0]
  order = [int(x) for x in order[0].split(" ")]
  tgts = tgts.split(" ")
  k = 0
  for i,x in enumerate(tgts):
    if x[0] == "<" and x[-1]=='>':
      tgts[i] = entlist[order[k]]
      k+=1
  return " ".join(tgts)
        
def test(args,ds,m,epoch='cmdline'):
  args.vbsz = 1
  model = args.save.split("/")[-1]
  m.eval()
  k = 0
  data = ds.mktestset(args)
  ofn = "outputs/"+model+".inputs.beam_predictions."+epoch
  ofng = "outputs/"+model+".inputs.beam_gt."+epoch
  pf = open(ofn,'w')
  pfg = open(ofng,'w')
  preds = []
  golds = []
  for b in data:
    #if k == 10: break
    print(k,len(data))
    b = ds.fixBatch(b)
    '''
    p,z = m(b)
    p = p[0].max(1)[1]
    gen = ds.reverse(p,b.rawent)
    '''
    gen = m.beam_generate(b,beamsz=4,k=6)
    gen.sort()
    gen = ds.reverse(gen.done[0].words,b.rawent)
    k+=1
    gold = ds.reverse(b.tgt[0][1:],b.rawent)
    preds.append(gen.lower())
    golds.append(gold.lower())
    #tf.write(ent+'\n')
    pf.write(gen.lower()+'\n')
    pfg.write(gold.lower() + '\n')
  pf.close()
  pfg.close()

  with open(ofn) as f:
    cands = {'generated_description' + str(i): x.strip() for i, x in enumerate(f.readlines())}
  with open(ofng) as f:
    refs = {'generated_description' + str(i): [x.strip()] for i, x in enumerate(f.readlines())}

  final_scores = evaluator.evaluate(live=True, cand=cands, ref=refs)
  logger.info("Results for model:\t", model_name)
  logger.info ('Bleu_1:\t', final_scores['Bleu_1'])
  logger.info ('Bleu_2:\t', final_scores['Bleu_2'])
  logger.info ('Bleu_3:\t', final_scores['Bleu_3'])
  logger.info ('Bleu_4:\t', final_scores['Bleu_4'])
  logger.info ('ROUGE_L:\t', final_scores['ROUGE_L'])
  logger.info ('METEOR:\t', final_scores['METEOR'])

  return preds,golds

'''
def metrics(preds,gold):
  cands = {'generated_description'+str(i):x.strip() for i,x in enumerate(preds)}
  refs = {'generated_description'+str(i):[x.strip()] for i,x in enumerate(gold)}
  x = evalMetrics.Evaluate()
  scores = x.evaluate(live=True, cand=cands, ref=refs)
  return scores
'''

if __name__=="__main__":
  args = pargs()
  args.eval = True
  ds = dataset(args)
  args = dynArgs(args,ds)
  m = model(args)
  m = m.to(args.device)
  models = glob.glob(args.save+"/*vloss*")
  m.args = args
  m.maxlen = args.max
  m.starttok = ds.OUTP.vocab.stoi['<start>']
  m.endtok = ds.OUTP.vocab.stoi['<eos>']
  m.eostok = ds.OUTP.vocab.stoi['.']
  args.vbsz = 1

  for i, model_name in enumerate(models):
    cpt = torch.load(model_name)
    m.load_state_dict(cpt)
    preds,gold = test(args,ds,m)
  '''
  scores = metrics(preds,gold)
  for k,v in scores.items():
    print(k+'\t'+str(scores[k]))
  '''
