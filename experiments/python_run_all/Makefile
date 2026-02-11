SHELL:= /bin/bash#
.SILENT: # no output

LOUD = \033[1;34m#
HIGH = \033[1;33m#
SOFT = \033[0m#

help: ## show help
	grep '^[a-z].*:.*##' $(MAKEFILE_LIST) \
	| sort \
	| gawk 'BEGIN {FS="##"; print "\n$(LOUD)make$(SOFT) [OPTIONS]\n"} \
	              {sub(/:.*/,"",$$1); \
                 printf("$(LOUD)%10s$(SOFT) %s\n",$$1,$$2)}'
	echo -e "$(HIGH)"; cat ../etc/frog.txt; echo -e "$(SOFT)"

vim: ## vim install
	mkdir -p ~/.vim
	if [[ ! -d ~/.vim/bundle ]]; \
	then git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim; \
	fi
	vim +'PluginInstall --sync' +qa

sh: ## run a shell
	bash --init-file  ../etc/dotshellrc -i

push: ## commit to main
	- echo -en "$(LOUD)Why this push? $(SOFT)" ;  read x ; git commit -am "$$x" ;  git push
	- git status

word: ## commit to main
	- echo -en "$(LOUD)phrase? $(SOFT)" ;  read x ; figlet -W -f mini $$x | gawk '{print "-- " $$0}' |pbcopy

../docs/%.html : ../src/%.md    
	1>&2 echo "... $@"
	sed -e '1d' -e '2d' $< | pandoc  -o $@ $(Pandoc)

~/tmp/%.html : %.py
	pydoc-markdown -p $^ > $@
	open $@

../docs/%.html : %.py
	pydoc-markdown -p $^ > $@
	gawk '/<h1>/ {print "<div class=docs>";                       \
                while(getline x < "../etc/head.html") {print x}; \
                print "<h1>'$^'</h1></div>";                  \
                next} 1' $@ > tmp.tmp
	mv tmp.tmp $@

~/tmp/%.pdf : %.py  Makefile
	@mkdir -p docs
	@echo "pdfing : $@ ... "
	@a2ps -Bj --landscape                           \
		--chars-per-line=90 \
		--line-numbers=1                    \
		--highlight-level=normal  \
		--columns 3                 \
		--borders=no --pro=color \
		--right-footer="" --left-footer=""    \
		--pretty-print=../etc/python.ssh             \
		--footer="page %p."                     \
		-M letter -o $@.ps $
	@ps2pdf $@.ps $@; rm $@.ps

heading:
	- echo -en "$(LOUD)Heading? $(SOFT)" ;  read x ; figlet -W -f mini $$x | gawk '{print "#  " $$0}'

comparez: ../../moot/optimize/[bcfhmprst]*/*.csv
	echo -n "" > comparez.out
	$(foreach f, $^, (python3 run_all.py --comparez $f >> $@.out& ); )

branch: ../../moot/optimize/[bcfhmprst]*/*.csv
	$(foreach f, $^, (python3 run_all.py --branch $f | sort -rn | fmt -100  &) ; )

branches:; $(MAKE) branch | tee ~/tmp/$@.out
comparezs:; $(MAKE) comparez | tee $@.out

report:
	@mkdir -p ../../results/optimization_performance

	# 1. Write header according to TASKS
	@echo "D,#R,#X,#Y,B4.mu,B4.lo,B4.sd,2B.mu, \
ACT-6,ACT-6_time,ACT-12,ACT-12_time,ACT-18,ACT-18_time,ACT-24,ACT-24_time,ACT-50,ACT-50_time,ACT-100,ACT-100_time,ACT-200,ACT-200_time, \
SMAC-6,SMAC-6_time,SMAC-12,SMAC-12_time,SMAC-18,SMAC-18_time,SMAC-24,SMAC-24_time,SMAC-50,SMAC-50_time,SMAC-100,SMAC-100_time,SMAC-200,SMAC-200_time, \
KM++-6,KM++-6_time,KM++-12,KM++-12_time,KM++-18,KM++-18_time,KM++-24,KM++-24_time,KM++-50,KM++-50_time,KM++-100,KM++-100_time,KM++-200,KM++-200_time, \
RAND-6,RAND-6_time,RAND-12,RAND-12_time,RAND-18,RAND-18_time,RAND-24,RAND-24_time,RAND-50,RAND-50_time,RAND-100,RAND-100_time,RAND-200,RAND-200_time, \
Before,Before_time,File" \
	> ../../results/optimization_performance/report_tmp.csv

	# 2. Append results from comparez.out
	egrep csv comparez.out >> ../../results/optimization_performance/report_tmp.csv

	# 3. Remove *_time & XPLOIT-like algos (not used anymore)
	awk 'BEGIN{FS=OFS=","} \
	NR==1 { \
	    for (i=1; i<=NF; i++) \
		if ($$i !~ /_time$$/) keep[i]=1; \
	} \
	{ \
	    first=1; \
	    for (i=1; i<=NF; i++) if (keep[i]) { \
		if (!first) printf OFS; \
		printf "%s", $$i; \
		first=0; \
	    } \
	    print ""; \
	}' ../../results/optimization_performance/report_tmp.csv \
	> ../../results/optimization_performance/report.csv

	# 4. Append % "a" columns summary (same as before)
	awk 'BEGIN{FS=OFS=","} \
	NR==1 { total=NF } \
	NR>1 { \
	    for (i=9; i<total; i++) { \
		N[i]++; if ($$i ~ /a$$/) A[i]++; \
	    } \
	} \
	END { \
	    printf ",,,,,,,,"; \
	    for (i=9; i<total; i++) { \
		p=int((A[i]?100*A[i]/N[i]:0)+0.5); \
		printf "%s%d", OFS, p; \
	    } \
	    printf ",\n"; \
	}' ../../results/optimization_performance/report.csv \
	>> ../../results/optimization_performance/report.csv



kill:
		ps | grep python | grep run_all | gawk '{system("kill -9 " $$1)}'