#
# Made by DanZimm on Mon Mar 27 13:16:29 CDT 2017
#
CXX ?= clang++
CC ?= clang
CFLAGS=-Wall -Werror -g
C++FLAGS=-std=c++14
OBJDIR=objects
load_labels_OBJ=$(addprefix $(OBJDIR)/, load_labels.o)
load_images_OBJ=$(addprefix $(OBJDIR)/, load_images.o)
tests_OBJ=$(addprefix $(OBJDIR)/, Tests.o MatrixTests.o Matrix.o)

.PHONY: all clean

all: $(OBJDIR) load_labels load_images

-include $(load_labels_OBJ:.o=.d)
-include $(load_images_OBJ:.o=.d)
-include $(tests_OBJ:.o=.d)

$(OBJDIR)/%.o: %.cc
	$(CXX) -c $< -o $@ $(CFLAGS) $(C++FLAGS)
	$(CXX) -MM -MT $@ $< $(CFLAGS) $(C++FLAGS) > $(OBJDIR)/$*.d

$(OBJDIR)/%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)
	$(CC) -MM -MT $@ $< $(CFLAGS) > $(OBJDIR)/$*.d

load_labels: $(load_labels_OBJ)
	$(CXX) $(filter %.o,$^) -o $@ $(CFLAGS) $(C++FLAGS)

load_images: $(load_images_OBJ)
	$(CXX) $(filter %.o,$^) -o $@ $(CFLAGS) $(C++FLAGS)

tests: $(OBJDIR) $(tests_OBJ)
	$(CXX) $(filter %.o,$^) -o $@ $(CFLAGS) $(C++FLAGS)

test: tests
	./tests

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) load_labels load_images tests

