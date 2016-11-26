CCP=clang++
CC=clang
CFLAGS=-Wall -Werror -g
C++FLAGS=-std=c++14
OBJDIR=objects
load_labels_OBJ=$(addprefix $(OBJDIR)/, load_labels.o)
load_images_OBJ=$(addprefix $(OBJDIR)/, load_images.o)

.PHONY: all clean

all: $(OBJDIR) load_labels load_images

-include $(load_labels_OBJ:.o=.d)

$(OBJDIR)/%.o: %.cc
	$(CCP) -c $< -o $@ $(CFLAGS) $(C++FLAGS)
	$(CCP) -MM -MT $@ $< $(CFLAGS) $(C++FLAGS) > $(OBJDIR)/$*.d

$(OBJDIR)/%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)
	$(CC) -MM -MT $@ $< $(CFLAGS) > $(OBJDIR)/$*.d

load_labels: $(load_labels_OBJ)
	$(CCP) $(filter %.o,$^) -o $@ $(CFLAGS) $(C++FLAGS)

load_images: $(load_images_OBJ)
	$(CCP) $(filter %.o,$^) -o $@ $(CFLAGS) $(C++FLAGS)

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) load_labels

