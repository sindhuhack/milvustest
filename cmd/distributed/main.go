// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"syscall"

	"github.com/milvus-io/milvus/cmd/distributed/roles"
)

const (
	RoleMaster       = "master"
	RoleProxyService = "proxyservice"
	RoleQueryService = "queryservice"
	RoleIndexService = "indexservice"
	RoleDataService  = "dataservice"
	RoleProxyNode    = "proxynode"
	RoleQueryNode    = "querynode"
	RoleIndexNode    = "indexnode"
	RoleDataNode     = "datanode"
	RoleMixture      = "mixture"
)

func getPidFileName(service string, alias string) string {
	var filename string
	if len(alias) != 0 {
		filename = fmt.Sprintf("%s-%s.pid", service, alias)
	} else {
		filename = service + ".pid"
	}
	return filename
}

func createPidFile(filename string, runtimeDir string) error {
	fileFullName := path.Join(runtimeDir, filename)

	fd, err := os.OpenFile(fileFullName, os.O_CREATE|os.O_RDWR, 0664)
	if err != nil {
		return fmt.Errorf("file %s is in-use, error = %w", filename, err)
	}
	fmt.Println("open pid file:", fileFullName)
	defer fd.Close()

	err = syscall.Flock(int(fd.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	if err != nil {
		return fmt.Errorf("file %s is in-use, error = %w", filename, err)
	}
	fmt.Println("lock pid file:", fileFullName)

	fd.Truncate(0)
	_, err = fd.WriteString(fmt.Sprintf("%d", os.Getpid()))

	return err
}

func closePidFile(fd *os.File) {
	fd.Close()
}

func removePidFile(filename string, runtimeDir string) {
	fileFullName := path.Join(runtimeDir, filename)
	os.Remove(fileFullName)
}

func run(role *roles.MilvusRoles, alias string, runtimeDir string) error {
	if role.EnableMaster {
		filename := getPidFileName(RoleMaster, alias)
		if err := createPidFile(filename, runtimeDir); err != nil {
			return fmt.Errorf("create pid file fail, service %s", filename)
		}
		defer removePidFile(filename, runtimeDir)
	}
	if role.EnableProxyService {
		filename := getPidFileName(RoleProxyService, alias)
		if err := createPidFile(filename, runtimeDir); err != nil {
			return fmt.Errorf("create pid file fail, service %s", filename)
		}
		defer removePidFile(filename, runtimeDir)
	}
	if role.EnableQueryService {
		filename := getPidFileName(RoleQueryService, alias)
		if err := createPidFile(filename, runtimeDir); err != nil {
			return fmt.Errorf("create pid file fail, service %s", filename)
		}
		defer removePidFile(filename, runtimeDir)
	}
	if role.EnableIndexService {
		filename := getPidFileName(RoleIndexService, alias)
		if err := createPidFile(filename, runtimeDir); err != nil {
			return fmt.Errorf("create pid file fail, service %s", filename)
		}
		defer removePidFile(filename, runtimeDir)
	}
	if role.EnableDataService {
		filename := getPidFileName(RoleDataService, alias)
		if err := createPidFile(filename, runtimeDir); err != nil {
			return fmt.Errorf("create pid file fail, service %s", filename)
		}
		defer removePidFile(filename, runtimeDir)
	}
	if role.EnableProxyNode {
		filename := getPidFileName(RoleProxyNode, alias)
		if err := createPidFile(filename, runtimeDir); err != nil {
			return fmt.Errorf("create pid file fail, service %s", filename)
		}
		defer removePidFile(filename, runtimeDir)
	}
	if role.EnableQueryNode {
		filename := getPidFileName(RoleQueryNode, alias)
		if err := createPidFile(filename, runtimeDir); err != nil {
			return fmt.Errorf("create pid file fail, service %s", filename)
		}
		defer removePidFile(filename, runtimeDir)
	}
	if role.EnableIndexNode {
		filename := getPidFileName(RoleIndexNode, alias)
		if err := createPidFile(filename, runtimeDir); err != nil {
			return fmt.Errorf("create pid file fail, service %s", filename)
		}
		defer removePidFile(filename, runtimeDir)
	}
	if role.EnableDataNode {
		filename := getPidFileName(RoleDataNode, alias)
		if err := createPidFile(filename, runtimeDir); err != nil {
			return fmt.Errorf("create pid file fail, service %s", filename)
		}
		defer removePidFile(filename, runtimeDir)
	}
	role.Run(false)
	return nil
}

func stopRole(filename string, runtimeDir string) error {
	var pid int

	fd, err := os.OpenFile(path.Join(runtimeDir, filename), os.O_RDONLY, 0664)
	// it's possible that pid file has already been removed
	if err != nil {
		return nil
	}
	defer closePidFile(fd)

	_, err = fmt.Fscanf(fd, "%d", &pid)
	if err != nil {
		return err
	}
	process, err := os.FindProcess(pid)
	// it's possible that this process has already been killed
	if err != nil {
		return nil
	}
	err = process.Signal(syscall.SIGTERM)
	if err != nil {
		return err
	}
	return nil
}

func stop(role *roles.MilvusRoles, alias string, runtimeDir string) error {
	if role.EnableMaster {
		filename := getPidFileName(RoleMaster, alias)
		if err := stopRole(filename, runtimeDir); err != nil {
			return fmt.Errorf("stop process fail, service %s", filename)
		}
	}
	if role.EnableProxyService {
		filename := getPidFileName(RoleProxyService, alias)
		if err := stopRole(filename, runtimeDir); err != nil {
			return fmt.Errorf("stop process fail, service %s", filename)
		}
	}
	if role.EnableQueryService {
		filename := getPidFileName(RoleQueryService, alias)
		if err := stopRole(filename, runtimeDir); err != nil {
			return fmt.Errorf("stop process fail, service %s", filename)
		}
	}
	if role.EnableIndexService {
		filename := getPidFileName(RoleIndexService, alias)
		if err := stopRole(filename, runtimeDir); err != nil {
			return fmt.Errorf("stop process fail, service %s", filename)
		}
	}
	if role.EnableDataService {
		filename := getPidFileName(RoleDataService, alias)
		if err := stopRole(filename, runtimeDir); err != nil {
			return fmt.Errorf("stop process fail, service %s", filename)
		}
	}
	if role.EnableProxyNode {
		filename := getPidFileName(RoleProxyNode, alias)
		if err := stopRole(filename, runtimeDir); err != nil {
			return fmt.Errorf("stop process fail, service %s", filename)
		}
	}
	if role.EnableQueryNode {
		filename := getPidFileName(RoleQueryNode, alias)
		if err := stopRole(filename, runtimeDir); err != nil {
			return fmt.Errorf("stop process fail, service %s", filename)
		}
	}
	if role.EnableIndexNode {
		filename := getPidFileName(RoleIndexNode, alias)
		if err := stopRole(filename, runtimeDir); err != nil {
			return fmt.Errorf("stop process fail, service %s", filename)
		}
	}
	if role.EnableDataNode {
		filename := getPidFileName(RoleDataNode, alias)
		if err := stopRole(filename, runtimeDir); err != nil {
			return fmt.Errorf("stop process fail, service %s", filename)
		}
	}
	return nil
}

func makeRuntimeDir(dir string) error {
	st, err := os.Stat(dir)
	if os.IsNotExist(err) {
		err = os.Mkdir(dir, 0755)
		if err != nil {
			return fmt.Errorf("create runtime dir %s failed", dir)
		}
		return nil
	}
	if !st.IsDir() {
		return fmt.Errorf("%s is not directory", dir)
	}
	tmpFile, err := ioutil.TempFile(dir, "tmp")
	if err != nil {
		return err
	}
	fileName := tmpFile.Name()
	tmpFile.Close()
	os.Remove(fileName)
	return nil
}

func main() {
	if len(os.Args) < 3 {
		_, _ = fmt.Fprint(os.Stderr, "usage: milvus [command] [server type] [flags]\n")
		return
	}
	command := os.Args[1]
	serverType := os.Args[2]
	flags := flag.NewFlagSet(os.Args[0], flag.ExitOnError)

	var svrAlias string
	flags.StringVar(&svrAlias, "alias", "", "set alias")

	var enableMaster, enableProxyService, enableQueryService, enableIndexService, enableDataService bool
	flags.BoolVar(&enableMaster, RoleMaster, false, "enable master")
	flags.BoolVar(&enableProxyService, RoleProxyService, false, "enable proxy service")
	flags.BoolVar(&enableQueryService, RoleQueryService, false, "enable query service")
	flags.BoolVar(&enableIndexService, RoleIndexService, false, "enable index service")
	flags.BoolVar(&enableDataService, RoleDataService, false, "enable data service")

	if err := flags.Parse(os.Args[3:]); err != nil {
		os.Exit(-1)
	}

	role := roles.MilvusRoles{}
	switch serverType {
	case RoleMaster:
		role.EnableMaster = true
	case RoleProxyService:
		role.EnableProxyService = true
	case RoleProxyNode:
		role.EnableProxyNode = true
	case RoleQueryService:
		role.EnableQueryService = true
	case RoleQueryNode:
		role.EnableQueryNode = true
	case RoleDataService:
		role.EnableDataService = true
	case RoleDataNode:
		role.EnableDataNode = true
	case RoleIndexService:
		role.EnableIndexService = true
	case RoleIndexNode:
		role.EnableIndexNode = true
	case RoleMixture:
		role.EnableMaster = enableMaster
		role.EnableProxyService = enableProxyService
		role.EnableQueryService = enableQueryService
		role.EnableDataService = enableDataService
		role.EnableIndexService = enableIndexService
	default:
		fmt.Fprintf(os.Stderr, "Unknown server type = %s\n", serverType)
		os.Exit(-1)
	}

	runtimeDir := "/run/milvus"
	if err := makeRuntimeDir(runtimeDir); err != nil {
		fmt.Fprintf(os.Stderr, "Set runtime dir at %s failed, set it to /tmp/milvus directory\n", runtimeDir)
		runtimeDir = "/tmp/milvus"
		if err = makeRuntimeDir(runtimeDir); err != nil {
			fmt.Fprintf(os.Stderr, "Create runtime directory at %s failed\n", runtimeDir)
			os.Exit(-1)
		}
	}

	switch command {
	case "run":
		if err := run(&role, svrAlias, runtimeDir); err != nil {
			panic(err)
		}
	case "stop":
		if err := stop(&role, svrAlias, runtimeDir); err != nil {
			panic(err)
		}
	default:
		fmt.Fprintf(os.Stderr, "unknown command : %s", command)
	}
}
