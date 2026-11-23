import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import time, config, mafia_core as core, mafia_graph as mg
t=time.time(); bad=0; days=[]; w={}
for seed in range(50):
    g=core.new_game(list(config.CHARACTERS),None,seed)
    ctr={p.name:core.RandomController(p.name,seed*10+i) for i,p in enumerate(g.players)}
    gr=mg.build_graph(ctr,on_event=lambda e:None)
    f=mg.run_spectator(gr,g,"t%d"%seed)
    assert f.winner, seed
    # compare with engine
    g2=core.new_game(list(config.CHARACTERS),None,seed)
    ctr2={p.name:core.RandomController(p.name,seed*10+i) for i,p in enumerate(g2.players)}
    e=core.GameEngine(g2,ctr2,on_event=lambda e:None); e.run()
    if (f.winner,f.day)!=(g2.winner,g2.day): bad+=1
    days.append(f.day); w[f.winner]=w.get(f.winner,0)+1
print("done",time.time()-t,"diff vs engine",bad,w,sum(days)/50)
gr=mg.build_graph(ctr)
print(mg.graph_mermaid(gr)[:300])
