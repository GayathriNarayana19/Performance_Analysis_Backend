**REQUIREMENTS**

```
sudo apt install python-is-python3 python3-pip python3-venv -y

python3 -m venv venv
source venv/bin/activate

pip install pandas pyyaml matplotlib PyPDF2
```
**STEPS TO EXECUTE**
```
python3 pmuv3_plotting.py -config config.yaml
```
**SAMPLE RUN**

You have a sample "bundles" folder. You can try running that as follows with config_test.yaml to get a hang of how this works. Later feel free to edit config.yaml
according to you project needs. 

This bundles folder has 15 bundles collected under two different contexts or for two different code segments. Hence, let us first split the CSVs as follows. 

NOTE: If you instrumented only for one code block you can ignore the below step as it has no context column in CSVs. 
```
python3 split_csvs.py path/to/bundle
```
You will see that a directory gets created and gets its name from the name of the directory you mention in the command. In the above case, it was "bundle". So bundle_split_files dir gets created which will have SECTION_1_split  and SECTION_2_split subdirectories each containing 15 CSVs. (This is because we have two contexts in this case - SECTION1, SECTION2)
```
python3 pmuv3_plotting.py -config config_test.yaml
```
The outputs will get stored in the directory path you mention. You can give the name of the desired dir in the config.yaml and it does the creation itself. In this sample run, only Section 1 has been plotted. 

**ADDITIONAL USEFUL INFORMATION FOR UNDERSTANDING**

If you would like to compare Section 1 context/code block on different processors like N1 and V1, you can do that as well. In such a scenario, if you named the directory as N1_bundle and V1_bundle, you would run, 
```
python3 split_csvs.py path/to/N1_bundle path/to/V1_bundle 

python3 pmuv3_plotting.py -config config_test.yaml
```
