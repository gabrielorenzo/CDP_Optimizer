from __future__ import print_function
import os
import sys
import subprocess

def analyze_cdp(imatest, image):
    from imatest.it import ImatestLibrary, ImatestException

    exit_code = 0
    root_dir = os.path.dirname(os.path.realpath(__file__))
    images_dir = r'C:\Users\F80lab1\Desktop\CDP_IM\Optimizer_CDP 1\Optimizer_CDP\optimizer\out'

    # Call to color_tone using Op Mode Standard, with ini file argument and JSON output
    # Calling the "color_tone_json" method returns the JSON result string only
    # All output files are written to the "Results" directory
    input_file = image
    op_mode = ImatestLibrary.OP_MODE_SEPARATE
    ini_file = os.path.join(root_dir, 'ini_file', r'CDP_inifile.ini')
    print(ini_file)
    try:
        result = imatest.color_tone(input_file=input_file,
                                   root_dir=images_dir,
                                   op_mode=op_mode,
                                   ini_file=ini_file)
    except ImatestException as iex:
        if iex.error_id == ImatestException.FloatingLicenseException:
            print("All floating license seats are in use.  Exit Imatest on another computer and try again.")
        elif iex.error_id == ImatestException.LicenseException:
            print("License Exception: " + iex.message)
        else:
            print(iex.message)

        exit_code = iex.error_id
    except Exception as ex:
        print(str(ex))
        exit_code = 1

    return exit_code

def main():
    from imatest.it import ImatestLibrary, ImatestException

    imatest = ImatestLibrary()
    exit_code = analyze(imatest)
    imatest.terminate_library()

    exit(exit_code)


if __name__ == "__main__":
    if sys.platform == 'darwin':
        if 'MWPYTHON_FORWARD' not in os.environ:
            file_path = os.path.abspath(__file__)

            command = ['/Applications/MATLAB/MATLAB_Runtime/R2023a/bin/mwpython', file_path]

            # Set an environment variable to halt recursion
            os.environ['MWPYTHON_FORWARD'] = '1'

            # Set the PYTHONHOME environment variable so that the mwpython script can correctly detect the
            # Python version
            os.environ['PYTHONHOME'] = sys.exec_prefix

            completed_process = subprocess.run(command, env=os.environ)

            exit(completed_process.returncode)
        else:
            main()
    else:
        main()