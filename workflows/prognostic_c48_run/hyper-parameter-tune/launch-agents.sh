SWEEP="$1"

envsubst SWEEP << EOF | kubectl create -f -
apiVersion: batch/v1
kind: Job
metadata:
  generateName: emulation-noah-sweep-
spec:
  parallelism: 6
  template:
    metadata:
      labels:
        app: emulation
    spec:
      volumes:
        - name: gcp-credentials-user-gcp-sa
          secret:
            secretName: gcp-key
        - name: dvc-cache
          emptyDir: {}
      containers:
        - name: main
          envFrom:
            - secretRef:
                name: wandb-token
          env:
            - name: GOOGLE_APPLICATION_CREDENTIALS
              value: /tmp/key
            - name: GOOGLE_APPLICATION_CREDENTIALS
              value: /secret/gcp-credentials/key.json
            - name: CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE
              value: /secret/gcp-credentials/key.json
          image:
            us.gcr.io/vcm-ml/emulator:4e6c7fe185219b827404d5303c892e0351d51812
          command: [wandb, agent, $SWEEP]
          volumeMounts:
            - name: gcp-credentials-user-gcp-sa
              mountPath: /secret/gcp-credentials
              readOnly: true
            - name: dvc-cache
              mountPath: /mnt/dvc
          resources:
            limits:
              cpu: "2000m"
              memory: 12Gi
              ephemeral-storage: 40Gi
            requests:
              cpu: "2000m"
              memory: 12Gi
              ephemeral-storage: 40Gi
      restartPolicy: Never
      tolerations:
        - effect: NoSchedule
          key: dedicated
          value: med-sim-pool
EOF