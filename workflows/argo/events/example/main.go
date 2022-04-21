// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// [START bigquery_quickstart]

// Sample bigquery-quickstart creates a Google BigQuery dataset.
package main

import (
	"cloud.google.com/go/bigquery"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"
)

const projectID = "vcm-ml"

type Ai2Labels struct {
	Author     string `json:"author"`
	Project    string `json:"project"`
	Experiment string `json:"experiment"`
	Trial      string `json:"trial"`
}

type Metadata struct {
	CreatedAt time.Time `json:"creationTimestamp"`
	Name      string    `json:"name"`
	Labels    json.RawMessage
	Uid       string `json:"uid"`
}

type Workflow struct {
	ApiVersion string   `json:"apiVersion"`
	Kind       string   `json:"kind"`
	Metadata   Metadata `json:"metadata"`
}

func (w *Workflow) GetLabels(labels *Output) error {
	// default labels
	labels.Author = "ai2cm"
	labels.Experiment = "default"
	labels.Project = "default"
	labels.Trial = w.Metadata.Name
	labels.Name = w.Metadata.Name
	labels.CreatedAt = w.Metadata.CreatedAt
	labels.Uid = w.Metadata.Uid

	return json.Unmarshal(w.Metadata.Labels, &labels)
}

type Output struct {
	Ai2Labels                   // makes Ai2Labes fields available here
	Labels    map[string]string `bigquery:"-"`
	Name      string
	CreatedAt time.Time
	Uid       string
	Contents  []byte
}

func UnmarshalOutput(data []byte, output *Output) error {
	var labels map[string]string = make(map[string]string)
	var w Workflow

	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}

	if w.Kind != "Workflow" {
		return fmt.Errorf("%s not a workflow.", w.Kind)
	}

	if err := json.Unmarshal(w.Metadata.Labels, &labels); err != nil {
		return err
	}

	if err := w.GetLabels(output); err != nil {
		return err
	} else {
		output.Contents = data
	}
	return nil
}

func dataToBq(dat []byte) error {
	var output Output

	ctx := context.Background()
	err := UnmarshalOutput(dat, &output)
	if err != nil {
		return err
	}

	schema, err := bigquery.InferSchema(Output{})
	if err != nil {
		return err
	}

	// Creates a client.
	client, err := bigquery.NewClient(ctx, projectID)
	if err != nil {
		return err
	}
	defer client.Close()

	// Sets the name for the new dataset.
	datasetName := "test_prog_logs"

	// Creates the new BigQuery dataset.
	ds := client.Dataset(datasetName)
	table := ds.Table("argo_workflows")
	err = table.Create(ctx, &bigquery.TableMetadata{
		ExpirationTime: time.Now().Add(1 * time.Hour)})
	if err != nil {
		// panic(err)
	}
	meta_updates := bigquery.TableMetadataToUpdate{Schema: schema}
	_, err = table.Update(ctx, meta_updates, "")
	if err != nil {
		return err
	}

	u := table.Inserter()
	items := []*Output{&output}

	if err := u.Put(ctx, items); err != nil {
		return err
	}
	log.Printf("Uploaded %s to big query", output.Uid)

	return nil
}

func extractBody(data []byte) ([]byte, error) {
	// make go struct to unback the "body" field
	var bodyContainer struct {
		Body string
	}

	if err := json.Unmarshal(data, &bodyContainer); err != nil {
		return nil, err
	}
	body := bodyContainer.Body
	bodyStr := []byte(body)
	return bodyStr, nil
}

func main() {

	http.HandleFunc("/bar", func(w http.ResponseWriter, r *http.Request) {

		log.Println("request received")

		data, err := io.ReadAll(r.Body)
		body, err := extractBody(data)
		if err != nil {
			log.Print(err)
			return
		}
		log.Print(string(body))
		if err != nil {
			log.Print(err)
			return
		}
		err = dataToBq(body)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, err.Error())
			log.Print(err)
		} else {
			w.WriteHeader(http.StatusOK)
			io.WriteString(w, "accepted")

		}
	})

	log.Fatal(http.ListenAndServe(":8080", nil))

	if len(os.Args) > 1 {
		file := os.Args[1]
		dat, err := os.ReadFile(file)
		if err != nil {
			log.Fatal(err)
		}
		dataToBq(dat)
	} else {
		log.Fatalf("No file specified")
	}

}
