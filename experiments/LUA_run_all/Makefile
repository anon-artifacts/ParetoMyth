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

~/tmp/%.html : %.lua
	pycco -d ~/tmp $^
	echo "p { text-align: right; }" >> ~/tmp/pycco.css
	open $@

../docs/%.html : %.lua
	pycco -d ../docs  $^
	echo "p { text-align: right;  }  " >> ../docs/pycco.css
	gawk '/<h1>/ {print "<div class=docs>";                       \
                while(getline x < "../etc/head.html") {print x}; \
                print "<h1>'$^'</h1></div>";                  \
                next} 1' $@ > tmp.tmp
	mv tmp.tmp $@

~/tmp/%.pdf : %.lua  Makefile
	@mkdir -p docs
	@echo "pdfing : $@ ... "
	@a2ps -Bj --landscape                           \
		--chars-per-line=90 \
		--line-numbers=1                    \
		--highlight-level=normal  \
		--columns 3                 \
		--borders=no --pro=color \
		--right-footer="" --left-footer=""    \
		--pretty-print=../etc/lua.ssh             \
		--footer="page %p."                     \
		-M letter -o $@.ps $<
	@ps2pdf $@.ps $@; rm $@.ps

heading:
	- echo -en "$(LOUD)Heading? $(SOFT)" ;  read x ; figlet -W -f mini $$x | gawk '{print "#  " $$0}'

comparez: ../../moot/optimize/[bcfhmprst]*/*.csv
	echo -n "" > comparez.out
	$(foreach f, $^, (lua run_all.lua --comparez $f >> $@.out& ); )

branch: ../../moot/optimize/[bcfhmprst]*/*.csv
	$(foreach f, $^, (lua run_all.lua --branch $f | sort -rn | fmt -100  &) ; )

branches:; $(MAKE) branch | tee ~/tmp/$@.out
comparezs:; $(MAKE) comparez | tee $@.out

report:
	@mkdir -p ../../results/optimization_performance
	@echo "D,#R,#X,#Y,B4.mu,B4.lo,B4.sd,2B.mu,DEHB-6,DEHB-6_sd,DEHB-6_time,DEHB-12,DEHB-12_sd,DEHB-12_time,DEHB-18,DEHB-18_sd,DEHB-18_time,DEHB-24,DEHB-24_sd,DEHB-24_time,DEHB-50,DEHB-50_sd,DEHB-50_time,DEHB-100,DEHB-100_sd,DEHB-100_time,DEHB-200,DEHB-200_sd,DEHB-200_time,NSGAIII-6,NSGAIII-6_sd,NSGAIII-6_time,NSGAIII-12,NSGAIII-12_sd,NSGAIII-12_time,NSGAIII-18,NSGAIII-18_sd,NSGAIII-18_time,NSGAIII-24,NSGAIII-24_sd,NSGAIII-24_time,NSGAIII-50,NSGAIII-50_sd,NSGAIII-50_time,NSGAIII-100,NSGAIII-100_sd,NSGAIII-100_time,NSGAIII-200,NSGAIII-200_sd,NSGAIII-200_time,SMAC-6,SMAC-6_sd,SMAC-6_time,SMAC-12,SMAC-12_sd,SMAC-12_time,SMAC-18,SMAC-18_sd,SMAC-18_time,SMAC-24,SMAC-24_sd,SMAC-24_time,SMAC-50,SMAC-50_sd,SMAC-50_time,SMAC-100,SMAC-100_sd,SMAC-100_time,SMAC-200,SMAC-200_sd,SMAC-200_time,TPE-6,TPE-6_sd,TPE-6_time,TPE-12,TPE-12_sd,TPE-12_time,TPE-18,TPE-18_sd,TPE-18_time,TPE-24,TPE-24_sd,TPE-24_time,TPE-50,TPE-50_sd,TPE-50_time,TPE-100,TPE-100_sd,TPE-100_time,TPE-200,TPE-200_sd,TPE-200_time,ACT-6,ACT-6_sd,ACT-6_time,ACT-12,ACT-12_sd,ACT-12_time,ACT-18,ACT-18_sd,ACT-18_time,ACT-24,ACT-24_sd,ACT-24_time,ACT-50,ACT-50_sd,ACT-50_time,ACT-100,ACT-100_sd,ACT-100_time,ACT-200,ACT-200_sd,ACT-200_time,EZR-6,EZR-6_sd,EZR-6_time,EZR-12,EZR-12_sd,EZR-12_time,EZR-18,EZR-18_sd,EZR-18_time,EZR-24,EZR-24_sd,EZR-24_time,EZR-50,EZR-50_sd,EZR-50_time,EZR-100,EZR-100_sd,EZR-100_time,EZR-200,EZR-200_sd,EZR-200_time,KM++-6,KM++-6_sd,KM++-6_time,RAND-6,RAND-6_sd,RAND-6_time,KM++-12,KM++-12_sd,KM++-12_time,RAND-12,RAND-12_sd,RAND-12_time,KM++-18,KM++-18_sd,KM++-18_time,RAND-18,RAND-18_sd,RAND-18_time,KM++-24,KM++-24_sd,KM++-24_time,RAND-24,RAND-24_sd,RAND-24_time,KM++-50,KM++-50_sd,KM++-50_time,RAND-50,RAND-50_sd,RAND-50_time,KM++-100,KM++-100_sd,KM++-100_time,RAND-100,RAND-100_sd,RAND-100_time,KM++-200,KM++-200_sd,KM++-200_time,RAND-200,RAND-200_sd,RAND-200_time,Before,Before_sd, File" > ../../results/optimization_performance/report_tmp.csv
	# 2. Append results from comparez.out
	egrep csv comparez.out >> ../../results/optimization_performance/report_tmp.csv

	# 3. Remove *_time & XPLOIT-like algos (not used anymore)
	awk 'BEGIN{FS=OFS=","} \
		NR==1 { \
			for (i=1; i<=NF; i++) { \
				if ($$i !~ /_time$$/ && $$i != "XPLOIT" && $$i != "XPLORE" && $$i != "ADAPT" && $$i != "SWAY") keep[i]=1 \
			} \
		} \
		{ \
			first = 1; \
			for (i=1; i<=NF; i++) if (keep[i]) { \
				if (!first) printf OFS; \
				printf "%s", $$i; \
				first=0 \
			} \
			print "" \
		}' ../../results/optimization_performance/report_tmp.csv > ../../results/optimization_performance/report.csv; \
		awk 'BEGIN{FS=OFS=","} \
		NR==1 { total_fields = NF } \
		NR>1 { \
			for (i=9; i<total_fields; i++) { \
				total[i]++; \
				if ($$i ~ /a$$/) count[i]++; \
			} \
		} \
		END { \
			printf ",,,,,,,,"; \
			for (i=9; i<total_fields; i++) { \
				pct = (count[i] ? int((100.0 * count[i] / total[i]) + 0.5) : 0); \
				printf "%s%d", (i>9 ? OFS : ""), pct; \
			} \
			printf ",\n" \
		}' ../../results/optimization_performance/report.csv >> ../../results/optimization_performance/report.csv



kill:
		ps | grep python | grep run_all | gawk '{system("kill -9 " $$1)}'
