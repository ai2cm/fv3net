


all-tendency-no-limit:
	./run_arch.sh all-tendency-no-limit.yaml linear all-tends-nolimit
	./run_arch.sh all-tendency-no-limit.yaml dense all-tends-nolimit
	./run_arch.sh all-tendency-no-limit.yaml rnn all-tends-nolimit

all-tendency-limited:
	./run_arch.sh all-tendency-limited.yaml linear all-tends-limited
	./run_arch.sh all-tendency-limited.yaml dense all-tends-limited
	./run_arch.sh all-tendency-limited.yaml rnn all-tends-limited

direct-cloud-limited:
	./run_arch.sh direct-cloud-limited.yaml linear direct-qc-limited
	./run_arch.sh direct-cloud-limited.yaml dense direct-qc-limited
	./run_arch.sh direct-cloud-limited.yaml rnn direct-qc-limited

all: all-tendency-no-limit all-tendency-limited direct-cloud-limited

.PHONY: all-tendency-no-limit all-tendency-limited direct-cloud-limited