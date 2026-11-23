import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import config, mafia_core as core, mafia_graph as mg, random
class Counting(core.RandomController):
    calls=0
    def night_action(self, view):
        Counting.calls+=1
        return super().night_action(view)
fails=0; tested=0
for seed in range(60):
    ai=random.Random(seed).sample(config.CHARACTERS,7)
    g=core.new_game(ai,config.HUMAN_NAME,seed)
    mafias=[p.name for p in g.players if p.role=="mafia"]
    if mafias[-1]!=config.HUMAN_NAME: continue
    first=mafias[0]; tested+=1
    ctr={p.name:core.RandomController(p.name,seed+i) for i,p in enumerate(g.players)}
    ctr[first]=Counting(first,seed)
    ctr[config.HUMAN_NAME]=mg.GraphHumanController(config.HUMAN_NAME)
    gr=mg.build_graph(ctr,on_event=lambda e:None)
    Counting.calls=0; nights=0
    def ans(p):
        global nights
        if p["kind"]=="kill":
            nights+=1
            alive_m=[x for x in p["alive"]]
        return "x" if p["kind"]=="speak" else p["candidates"][0]
    f=mg.run_with_human(gr,g,"s%d"%seed,answer_fn=ans)
    # 첫 마피아(AI)는 밤마다 정확히 1번만 호출돼야 한다 (사람 interrupt 재개로 재호출되면 일수보다 많아진다)
    if Counting.calls>f.day: fails+=1
    assert f.winner
print("human-second-mafia games",tested,"first-mafia extra calls anomalies",fails)
