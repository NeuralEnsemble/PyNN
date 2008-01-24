"""
Script to automatically install NEURON with Python support.
Requires a version of locate that supports the -r option
"""

import urllib, tempfile, subprocess, os, re, datetime, sys

red     = 0010; green  = 0020; yellow = 0030; blue = 0040;
magenta = 0050; cyan   = 0060; bright = 0100
try:
    import ll.ansistyle   
    def colour(col,text):
        return str(ll.ansistyle.Text(col,str(text)))
except ImportError:
    def colour(col,text):
            return text

fullname = {'nrn': 'NEURON', 'iv': 'Interviews'}

#install_dir = tempfile.mkdtemp()
#install_dir = "/tmp/tmpjqWBgE"
install_dir = os.getcwd()
current_packages = {'iv': 'iv-17.tar.gz',
                    'nrn': 'nrn-5.9.tar.gz'} #'alpha/nrn-6.0.alpha-855.tar.gz'}
#base_url = "http://www.neuron.yale.edu/ftp/neuron/versions/alpha/"
base_url = "http://www.neuron.yale.edu/ftp/neuron/versions/v5.9/"

version_pattern1 = re.compile(r'NEURON -- Version (?P<version>\d\.\d) (?P<date>\d\d\d\d-\d\d?-\d\d?) .* Main \((?P<revision>\d+)\)')
version_pattern2 = re.compile(r'NEURON -- Release (?P<version>\d.\d.\d) \((?P<revision>\d+)\) (?P<date>\d\d\d\d-\d\d?-\d\d?)')

def search_for_neuron():
    """
    Search for existing NEURON distributions that have been compiled with the
    --with-nrnpython flag.
    """
    # We search for nrnivmodl in order to find nrniv, since nrnivmodl is never
    # a directory name
    p = subprocess.Popen('locate -r nrnivmodl$', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    found = [path.replace('nrnivmodl','nrniv') for path in set(p.stdout.read().split())]
    valid_installations = {}
    for nrniv_path in found:
        if os.path.exists(nrniv_path):
            cmd = "%s --version" % nrniv_path
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
            version = p.stdout.read().strip("\n")
            match = version_pattern1.search(version) or version_pattern2.search(version)
            if match:
                version = match.groupdict()
            errors = p.stderr.read()
            if version and not errors:
                cmd = "%s -python" % nrniv_path
                p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
                output, errors = p.communicate('import hoc')
                if not "syntax error" in errors:
                    valid_installations[nrniv_path] = version
    return valid_installations

def search_for_iv():
    """
    Return the path to the most recently created iv installation.
    """
    iv_path = None
    p = subprocess.Popen("locate -r 'lib/libIVhines.so.[0-9].[0-9].[0-9]'", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    found = ["/".join(path.split('/')[:-3]) for path in set(p.stdout.read().split())]
    if found:
        latest_time = 0
        for path in found:
            creation_time = os.path.getctime(path)
            if creation_time > latest_time:
                latest_time = creation_time
                iv_path = path
    return iv_path

def select_most_recent(installation_dict):
    """
    Return the path to the most recent of the NEURON installations found by
    search_for_neuron().
    """
    nrniv_path = ''
    latest_date = datetime.date(1970,1,1)
    for path, version_dict in installation_dict.items():
        date = datetime.date(*[int(x) for x in version_dict['date'].split('-')])
        if date > latest_date:
            latest_date = date
            nrniv_path = path
    return nrniv_path

def download_and_install(pkg,extra_options=''):
    """
    Download and install NEURON or Interviews.
    """
    assert pkg in ('iv','nrn')
    print colour(bright+blue, "Downloading %s" % fullname[pkg])
    urllib.urlretrieve("%s/%s" % (base_url,current_packages[pkg]), install_dir+"/%s" % current_packages[pkg])
    cmd = """
            dir=%s
            p=%s
            cd $dir
            tar xzf $p-*.tar.gz
            mv $p-*[0-9] $p
          """ % (install_dir,pkg)
    print cmd
    os.system(cmd)
    
    print colour(bright+blue, "Building %s" % fullname[pkg])
    if pkg == 'nrn':
        extra_options += " --with-nrnpython"
    cmd = """
            dir=%s/%s
            cd $dir
            ./configure --prefix=`pwd` %s > config.out
            make > make.log 2> make.errors
          """ % (install_dir,pkg,extra_options)
    print cmd
    retval = os.system(cmd)
    if retval != 0: # error during compilation
        print colour(red,"Compilation errors: ")
        f = open('%s/%s/make.errors' % (install_dir,pkg),'r'); errors = f.read(); f.close()
        if errors:
            print colour(red, errors)
        sys.exit(2) # should probably try to clean up first
        
    print colour(bright+blue, "Installing %s in %s/%s" % (fullname[pkg],install_dir,pkg))
    cmd = """
            dir=%s/%s
            cd $dir
            make install > install.log 2> install.errors
          """ % (install_dir,pkg)
    print cmd
    retval = os.system(cmd)
    if retval != 0: # error during installation
        print colour(red,"Installation errors: ")
        f = open('%s/%s/install.errors' % (install_dir,pkg),'r'); errors = f.read(); f.close()
        if errors:
            print colour(red, errors)
            sys.exit(2) # should probably try to clean up first
    return "%s/%s" % (install_dir,pkg)

# ==============================================================================
if __name__ == "__main__":
    # Search for existing NEURON installations that are python-enabled and
    # sufficiently up-to-date for PyNN
    existing_nrn_installations = search_for_neuron()
    if existing_nrn_installations:
       nrniv_path = select_most_recent(existing_nrn_installations)
       print colour(bright+blue,"Using existing NEURON installation at %s" % nrniv_path)
    else:
       # If no suitable NEURON installations are found, check for an existing Interviews installation
        iv_path = search_for_iv()
        if iv_path:
            print colour(bright+blue,"Using existing Interviews installation at %s" % iv_path)
        else:
            # If not found, download and install Interviews in the current directory
            iv_path = download_and_install('iv')
        # Download and install NEURON --with-nrnpython
        nrniv_path = download_and_install('nrn','--with-iv=%s' % iv_path)
    
    # On Unix systems, set alias to nrnpython, or create shell script?
    run_nrnivmodl() # cd ../hoc; ../misc/nrn/<arch>/bin/nrnivmodl
    set_alias() # alias nrnpython='/home/andrew/Projects/FACETS/PyNN/pyNN/hoc/i686/special -python'