package main

import (
	"milvus_go_test/utils"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/milvus"
	"github.com/stretchr/testify/assert"
)

func TestInsert(t *testing.T) {
	client, name := Collection(true, milvus.VECTORFLOAT)
	insertParam := milvus.InsertParam{
		name,
		GenDefaultFieldValues(milvus.VECTORFLOAT),
		nil,
		""}
	ids, status, _ := client.Insert(insertParam)
	// t.Log(ids)
	t.Log(status)
	assert.Equal(t, status.Ok(), true)
	assert.Equal(t, len(ids), utils.DefaultNb)
}

func TestInsertWithCustomIds(t *testing.T) {
	client, name := Collection(false, milvus.VECTORFLOAT)
	var customIds []int64 = utils.DefaultIntValues
	insertParam := milvus.InsertParam{
		name,
		GenDefaultFieldValues(milvus.VECTORFLOAT),
		customIds,
		""}
	ids, status, _ := client.Insert(insertParam)
	// t.Log(ids)
	t.Log(status)
	assert.Equal(t, status.Ok(), true)
	assert.Equal(t, len(ids), utils.DefaultNb)
	assert.Equal(t, ids, customIds)
}

func TestInsertWithCustomIdsNotMatch(t *testing.T) {
	client, name := Collection(false, milvus.VECTORFLOAT)
	var customIds []int64 = utils.GenDefaultIntValues(utils.DefaultNb - 1)
	insertParam := milvus.InsertParam{
		name,
		GenDefaultFieldValues(milvus.VECTORFLOAT),
		customIds,
		""}
	ids, status, _ := client.Insert(insertParam)
	// t.Log(ids)
	t.Log(status)
	assert.Equal(t, status.Ok(), false)
	assert.Equal(t, len(ids), 0)
}
