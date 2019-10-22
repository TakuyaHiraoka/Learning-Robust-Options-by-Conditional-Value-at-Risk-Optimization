# Learning-Robust-Options-by-Conditional-Value-at-Risk-Optimization
The source code to replicate our NeurIPS 2019 paper ([arXiv preprint](https://arxiv.org/abs/1905.09191)). 
[Demo video](https://drive.google.com/open?id=1DRmIaK5VomCey70rKD_5DgX2Jm_1rFlo)

## Getting Started
1. Install [anaconda](https://www.anaconda.com/)  
2. Intsall [Mujoco](http://www.mujoco.org/) (required mjpro-150)  
3. Create conda environment containing required packages (e.g., [gym](https://github.com/openai/gym)):  
``` conda env create -f robustoption20190919.yml ```  
4. Replace the original "gym" directory installed in the conda environment with our version (the "gym" directory in our package).  
5. Install "gym-extensions" in our package.  
``` cd gym_extensions ```  
``` python setup.py intall ```

## Usage
1. Edit "main" function in "LearningMoreRobustOption/run_mujodo.py" according to an experiment setup (e.g., task environment and learning method). 
For example, if you  want to learn options for "HaflCheetah-disc", set the default value of python argument "--env" as :  
``` parser.add_argument('--env', help='environment ID', default='HalfCheetah-Random-Params-discrete-v1') ```  
You can select an option learning method by editing the default value of the python argument "--method." For example, if you want to use OC3, edit the source code as :  
``` parser.add_argument('--method', help='Method name:' + str(METHODS), type=str, default="CVaR") ```  
2. Run a script to conduct option learning:  
```sh runexp.sh```  
3. Run a script to select learned options to be tested:  
```python GeneratebestpolTextMaxAverageReturnwithCVaRThreth.py```  
or  
```python GeneratebestpolTextMaxAverageReturn.py```  
4. Run a script to conduct test:  
```sh run_test_w_best_cvar_pol.sh```  
5. Run a result summarizer. 
You can obtain the summary of CVaR scores as  
```python EvalAverageCVaR.py```  
and the summary of average return as  
```python EvalAverageReturn.py```  

## Acknowledgements
** This repository is based on [PPOC](https://github.com/mklissa/PPOC), [gym](https://github.com/openai/gym), and [gym extensions](https://github.com/Breakend/gym-extensions). **

## TODO
Keep refactoring the source codes. 