from inductor import BartInductor

inductor = BartInductor(prompt=False, ssts=True, dpp=True)

relation1 = '<mask> is a city of <mask>.'
relation2 = '<mask> is a member of <mask>.'
relation3 = '<mask> is plays in <mask> position.'
premise1 = 'his guitar work on the title track is credited as what first drew <mask> to him , who two years later invited allman to join him as part of <mask> .'
premise2 = 'kane \'s ultimate version was introduced in \" ultimate x - men \" # 76 , where he is once again a partner to <mask> , domino and the other <mask> team members .'
premise3 = 'in 2009 to mark the 125th anniversary of the <mask> he was named by the irish news as one of the all - time best 125 footballers from <mask> .'
score1 = inductor.score_premise_through_ins(inductor.generate_ins_from_hypo(relation1), relation1, premise1)
score2 = inductor.score_premise_through_ins(inductor.generate_ins_from_hypo(relation2), relation2, premise1)
score3 = inductor.score_premise_through_ins(inductor.generate_ins_from_hypo(relation3), relation3, premise1)
score4 = inductor.score_premise_through_ins(inductor.generate_ins_from_hypo(relation1), relation1, premise2)
score5 = inductor.score_premise_through_ins(inductor.generate_ins_from_hypo(relation2), relation2, premise2)
score6 = inductor.score_premise_through_ins(inductor.generate_ins_from_hypo(relation3), relation3, premise2)
score7 = inductor.score_premise_through_ins(inductor.generate_ins_from_hypo(relation1), relation1, premise3)
score8 = inductor.score_premise_through_ins(inductor.generate_ins_from_hypo(relation2), relation2, premise3)
score9 = inductor.score_premise_through_ins(inductor.generate_ins_from_hypo(relation3), relation3, premise3)
print(score1, score2, score3)
print(score4, score5, score6)
print(score7, score8, score9)