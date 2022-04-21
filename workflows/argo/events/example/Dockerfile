# syntax=docker/dockerfile:1

FROM golang:1.16-alpine

WORKDIR /app

COPY go.mod ./
COPY go.sum ./
COPY *.go ./

RUN go mod download
RUN go build -o /bin/main
