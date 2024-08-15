from msr_debugger import *

if __name__ == "__main__":
    tol=0.001#0.0001 #1000
    eta=0.01#0.01 #10000
    number_of_sites=8

    print("\n\n\n###########################Running MSR CSV Debugger (javier-jess data) #################################\n\n")
    print(f'eta: {eta}, tol: {tol}')
    print(f'number_of_sites: {number_of_sites}')
    test_path= '../test/jess_input_debug/allFiles/'
    #test_path= '../test/javier_jess_data/allFiles/'
    print(f'Data path: {test_path}')
    #run_msr_fed(tol=tol, eta=eta, number_of_sites=number_of_sites, data_dir = test_path)
    #run_msr_pooled(tol=tol, eta=eta, number_of_sites=number_of_sites, data_dir =test_path)
    all_pooled_OLS(data_dir=test_path)

    print(f'X_labels: {X_labels}')
    print(f'y_labels: {y_labels}')
