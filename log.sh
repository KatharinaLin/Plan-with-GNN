#python GNN-learn.py 2 50 1.0 100 1 > zeno-n50-p100.txt
#python plan.py 2 50 1.0 100 1 > zeno-plan-n50-p100.txt
#python GNN-learn.py 2 50 0.8 80 1 > zeno-n50-p80.txt
#python plan.py 2 50 0.8 80 1 > zeno-plan-n50-p80.txt
#python GNN-learn.py 2 50 0.6 60 1 > zeno-n50-p60.txt
#python plan.py 2 50 0.8 60 1 > zeno-plan-n50-p60.txt
#python GNN-learn.py 2 50 0.4 40 1 > zeno-n50-p40.txt
#python plan.py 2 50 0.4 40 1 > zeno-plan-n50-p40.txt
#python GNN-learn.py 2 50 0.2 20 1 > zeno-n50-p20.txt
#python plan.py 2 50 0.2 20 1 > zeno-plan-n50-p20.txt

#python GNN-learn.py 2 100 0.8 80 1 > zeno-n100-p80.txt
#python plan.py 2 100 0.8 80 1 > zeno-plan-n100-p80.txt
#python GNN-learn.py 2 100 0.4 40 1 > zeno-n100-p40.txt
#python plan.py 2 100 0.4 40 1 > zeno-plan-n100-p40.txt
#python GNN-learn.py 2 100 0.2 20 1 > zeno-n100-p20.txt
#python plan.py 2 100 0.2 20 1 > zeno-plan-n100-p20.txt
#python GNN-learn.py 0 50 0.2 20 1 > log-n50-p20.txt
#python plan.py 0 50 0.2 20 1 > log-plan-n50-p20.txt
#python GNN-learn.py 0 50 0.4 40 1 > log-n50-p40.txt
#python plan.py 0 50 0.4 40 1 > log-plan-n50-p40.txt
#python GNN-learn.py 0 50 0.6 60 1 > log-n50-p60.txt
#python plan.py 0 50 0.6 60 1 > log-plan-n50-p60.txt
#python GNN-learn.py 0 50 0.8 80 1 > log-n50-p80.txt
#python plan.py 0 50 0.8 80 1 > log-plan-n50-p80.txt
#python GNN-learn.py 0 50 1.0 100 1 > log-n50-p100.txt
#python plan.py 0 50 1.0 100 1 > log-plan-n50-p100.txt
#python RNN_learn.py 1 2000 0.6 60 2001 50 0 > dep-n2000-p60.txt
#python RNN_learn.py 1 2000 0.8 80 2001 50 0 > dep-n2000-p80.txt
#python RNN_learn.py 0 2000 0.6 60 2001 50 0 > log-n2000-p60.txt
#python RNN_learn.py 0 2000 0.8 80 2001 50 0 > log-n2000-p80.txt
#python RNN_learn.py 1 2000 0.6 60 2001 100 0 > dep-n2000-p60.txt
#python RNN_learn.py 1 2000 0.8 80 2001 100 0 > dep-n2000-p80.txt
#python new_plan.py 0 2000 0.6 60 2001 100 0 > log-plan-n2000-p60.txt
#python RNN_learn.py 2 2000 0.6 60 2001 100 0 > zeno-n2000-p60.txt
#python RNN_learn.py 2 2000 0.4 40 2001 100 0 > zeno-n2000-p40.txt
#python RNN_learn.py 2 2000 0.2 20 2001 100 0 > zeno-n2000-p20.txt
# python RNN_learn.py 4 2000 0.8 80 2001 100 0 > ferry-n2000-p80.txt
# python new_plan.py 4 2000 0.8 80 2001 100 0 > ferry-plan-n2000-p80.txt
#python new_plan.py 0 2000 0.4 40 2001 100 0 > log-plan-n2000-p40.txt
#python new_plan.py 0 2000 0.2 20 2001 100 0 > log-plan-n2000-p20.txt
#python new_plan.py 1 2000 0.2 20 2001 100 0 > dep-plan-n2000-p20.txt
#python new_plan.py 1 2000 0.4 40 2001 100 0 > dep-plan-n2000-p40.txt
# python new_plan.py 3 2000 0.2 20 2001 100 0 > sat-plan-n2000-p20.txt
# python new_plan.py 3 2000 0.4 40 2001 100 0 > sat-plan-n2000-p40.txt
# python new_plan.py 3 2000 0.6 60 2001 100 0 > sat-plan-n2000-p60.txt
#python RNN_learn.py 3 2000 0.8 80 2001 100 0 > sat-n2000-p80.txt
#python new_plan.py 3 2000 0.8 80 2001 100 0 > sat-plan-n2000-p80.txt
# python RF.py 2 2000 0.2 20 2001 100 0 > zeno-plan-rf-n2000-p20.txt
# python RF.py 2 2000 0.4 40 2001 100 0 > zeno-plan-rf-n2000-p40.txt
# python RF.py 2 2000 0.6 60 2001 100 0 > zeno-plan-rf-n2000-p60.txt
# python RF.py 2 2000 0.8 80 2001 100 0 > zeno-plan-rf-n2000-p80.txt

python RNN_learn.py 0 2000 0.8 80 2001 100 1 20 > d20.txt
python RNN_learn.py 0 2000 0.8 80 2001 100 1 40 > d40.txt
python RNN_learn.py 0 2000 0.8 80 2001 100 1 60 > d60.txt
python RNN_learn.py 0 2000 0.8 80 2001 100 1 80 > d80.txt
python RNN_learn.py 0 2000 0.8 80 2001 100 1 100 > d100.txt
# python new_plan.py 6 200 1.0 100 201 50 0 > grid-plan-n200-p80.txt

# python RNN_learn.py 4 2000 0.6 60 2001 100 0 > ferry-n2000-p60.txt
# python new_plan.py 4 2000 0.6 60 2001 100 0 > ferry-plan-n2000-p60.txt
# python RNN_learn.py 4 2000 0.4 40 2001 100 0 > ferry-n2000-p40.txt
# python new_plan.py 4 2000 0.4 40 2001 100 0 > ferry-plan-n2000-p40.txt
# python RNN_learn.py 4 2000 0.2 20 2001 100 0 > ferry-n2000-p20.txt
# python new_plan.py 4 2000 0.2 20 2001 100 0 > ferry-plan-n2000-p20.txt
# python RNN_learn.py 4 2000 1.0 100 2001 100 0 > ferry-n2000-p100.txt
# python new_plan.py 4 2000 1.0 100 2001 100 0 > ferry-plan-n2000-p100.txt
# python svm.py 4 2000 0.2 20 2001 100 1 > ferry-plan-svm-n2000-p20.txt
# python svm.py 4 2000 0.4 40 2001 100 1 > ferry-plan-svm-n2000-p40.txt
# python svm.py 4 2000 0.6 60 2001 100 1 > ferry-plan-svm-n2000-p60.txt
# python svm.py 4 2000 0.8 80 2001 100 1 > ferry-plan-svm-n2000-p80.txt
# python svm.py 4 2000 1.0 100 2001 100 1 > ferry-plan-svm-n2000-p100.txt
# python RF.py 4 2000 0.2 20 2001 100 1 > ferry-plan-rf-n2000-p20.txt
# python RF.py 4 2000 0.4 40 2001 100 1 > ferry-plan-rf-n2000-p40.txt
# python RF.py 4 2000 0.6 60 2001 100 0 > ferry-plan-rf-n2000-p60.txt
# python RF.py 4 2000 0.8 80 2001 100 0 > ferry-plan-rf-n2000-p80.txt
# python RF.py 4 2000 1.0 100 2001 100 0 > ferry-plan-rf-n2000-p100.txt
# python RNN_learn.py 6 2000 0.8 80 2001 100 1 > grid-n2000-p80.txt
# python new_plan.py 6 2000 0.8 80 2001 100 1 > grid-plan-n2000-p80.txt

# python RNN_learn.py 6 2000 0.6 60 2001 100 1 > grid-n2000-p60.txt
# python new_plan.py 6 2000 0.6 60 2001 100 1 > grid-plan-n2000-p60.txt
# python RNN_learn.py 6 2000 0.4 40 2001 100 1 > grid-n2000-p40.txt
# python new_plan.py 6 2000 0.4 40 2001 100 1 > grid-plan-n2000-p40.txt
# python RNN_learn.py 6 2000 0.2 20 2001 100 1 > grid-n2000-p20.txt
# python new_plan.py 6 2000 0.2 20 2001 100 1 > grid-plan-n2000-p20.txt
# python RNN_learn.py 6 2000 1.0 100 2001 100 1 > grid-n2000-p100.txt
# python new_plan.py 6 2000 1.0 100 2001 100 1 > grid-plan-n2000-p100.txt
# python plans.py 6 2000 0.4 40 2001 100 1 > grid-plan-pre-n2000-p40.txt
# python plans.py 6 2000 0.2 20 2001 100 1 > grid-plan-pre-n2000-p20.txt
# python plans.py 6 2000 0.6 60 2001 100 1 > grid-plan-pre-n2000-p60.txt
# python plans.py 6 2000 0.8 80 2001 100 1 > grid-plan-pre-n2000-p80.txt
# # python plans.py 6 2000 1.0 100 2001 100 1 > grid-plan-pre-n2000-p100.txt
# # python svm.py 6 2000 0.2 20 2001 100 1 > grid-plan-svm-n2000-p20.txt
# python svm.py 6 2000 0.4 40 2001 100 1 > grid-plan-svm-n2000-p40.txt
# python svm.py 6 2000 0.6 60 2001 100 1 > grid-plan-svm-n2000-p60.txt
# python svm.py 6 2000 0.8 80 2001 100 1 > grid-plan-svm-n2000-p80.txt
# python svm.py 6 2000 1.0 100 2001 100 1 > grid-plan-svm-n2000-p100.txt
# # python RF.py 6 2000 0.2 20 2001 100 1 > grid-plan-rf-n2000-p20.txt
# # python RF.py 6 2000 0.4 40 2001 100 1 > grid-plan-rf-n2000-p40.txt
# # python RF.py 6 2000 0.6 60 2001 100 1 > grid-plan-rf-n2000-p60.txt
# # python RF.py 6 2000 0.8 80 2001 100 1 > grid-plan-rf-n2000-p80.txt
# # python RF.py 6 2000 1.0 100 2001 100 1 > grid-plan-rf-n2000-p100.txt
# python svm.py 5 2000 0.2 20 2001 100 1 > mp-plan-svm-n2000-p20.txt
# python svm.py 5 2000 0.4 40 2001 100 1 > mp-plan-svm-n2000-p40.txt
# python svm.py 5 2000 0.6 60 2001 100 1 > mp-plan-svm-n2000-p60.txt
# python svm.py 5 2000 0.8 80 2001 100 1 > mp-plan-svm-n2000-p80.txt
# python svm.py 5 2000 1.0 100 2001 100 1 > mp-plan-svm-n2000-p100.txt
# python RF.py 5 2000 0.2 20 2001 100 1 > mp-plan-rf-n2000-p20.txt
# python RF.py 5 2000 0.4 40 2001 100 1 > mp-plan-rf-n2000-p40.txt
# python RF.py 5 2000 0.6 60 2001 100 1 > mp-plan-rf-n2000-p60.txt
# python RF.py 5 2000 0.8 80 2001 100 1 > mp-plan-rf-n2000-p80.txt
# python RF.py 5 2000 1.0 100 2001 100 1 > mp-plan-rf-n2000-p100.txt
#python 0-plan.py 2 2000 0.2 20 2001 100 1 > zeno-plan-n2000-p0.txt
#python 0-plan.py 1 2000 0.2 20 2001 100 1 > dep-plan-n2000-p0.txt
#python 0-plan.py 0 2000 0.2 20 2001 100 1 > log-plan-n2000-p0.txt
