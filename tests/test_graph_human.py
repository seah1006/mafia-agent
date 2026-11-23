import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import config, mafia_core as core, mafia_graph as mg, random, warnings
warnings.simplefilter("error")
kinds=[]
bad=0
for seed in range(20):
    ai=random.Random(seed).sample(config.CHARACTERS,7)
    g=core.new_game(ai,config.HUMAN_NAME,seed)
    ctr={p.name:core.RandomController(p.name,seed+i) for i,p in enumerate(g.players)}
    ctr[config.HUMAN_NAME]=mg.GraphHumanController(config.HUMAN_NAME)
    ev=[]
    gr=mg.build_graph(ctr,on_event=ev.append)
    def ans(p):
        kinds.append(p["kind"])
        assert p["player"]==config.HUMAN_NAME
        if p["kind"]=="speak": return "저는 시민입니다."
        return p["candidates"][0]
    f=mg.run_with_human(gr,g,"h%d"%seed,answer_fn=ans)
    assert f.winner
    h=[e for e in ev if e.get("visibility")=="secret"]
    if h: bad+=1
print("ok", {k:kinds.count(k) for k in set(kinds)}, "secret leaks",bad)
