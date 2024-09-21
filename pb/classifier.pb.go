// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.25.0-devel
// 	protoc        v3.14.0
// source: pb/classifier.proto

package pb

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// Classifier Message.
type Classifier struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Learned     float64   `protobuf:"fixed64,1,opt,name=learned,proto3" json:"learned,omitempty"`
	MinProb     float64   `protobuf:"fixed64,2,opt,name=minProb,proto3" json:"minProb,omitempty"`
	ClassInt32  []int32   `protobuf:"varint,4,rep,packed,name=ClassInt32,proto3" json:"ClassInt32,omitempty"`
	ClassInt64  []int64   `protobuf:"varint,5,rep,packed,name=ClassInt64,proto3" json:"ClassInt64,omitempty"`
	ClassUint32 []uint32  `protobuf:"varint,6,rep,packed,name=ClassUint32,proto3" json:"ClassUint32,omitempty"`
	ClassUint64 []uint64  `protobuf:"varint,7,rep,packed,name=ClassUint64,proto3" json:"ClassUint64,omitempty"`
	ClassFloat  []float32 `protobuf:"fixed32,8,rep,packed,name=ClassFloat,proto3" json:"ClassFloat,omitempty"`
	ClassDouble []float64 `protobuf:"fixed64,9,rep,packed,name=ClassDouble,proto3" json:"ClassDouble,omitempty"`
	ClassString []string  `protobuf:"bytes,10,rep,name=ClassString,proto3" json:"ClassString,omitempty"`
	Prior       []float64 `protobuf:"fixed64,11,rep,packed,name=prior,proto3" json:"prior,omitempty"`
	Prob        []*Prob   `protobuf:"bytes,12,rep,name=prob,proto3" json:"prob,omitempty"`
	ValueInt32  []int32   `protobuf:"varint,13,rep,packed,name=ValueInt32,proto3" json:"ValueInt32,omitempty"`
	ValueInt64  []int64   `protobuf:"varint,14,rep,packed,name=ValueInt64,proto3" json:"ValueInt64,omitempty"`
	ValueUint32 []uint32  `protobuf:"varint,15,rep,packed,name=ValueUint32,proto3" json:"ValueUint32,omitempty"`
	ValueUint64 []uint64  `protobuf:"varint,16,rep,packed,name=ValueUint64,proto3" json:"ValueUint64,omitempty"`
	ValueFloat  []float32 `protobuf:"fixed32,17,rep,packed,name=ValueFloat,proto3" json:"ValueFloat,omitempty"`
	ValueDouble []float64 `protobuf:"fixed64,18,rep,packed,name=ValueDouble,proto3" json:"ValueDouble,omitempty"`
	ValueString []string  `protobuf:"bytes,19,rep,name=ValueString,proto3" json:"ValueString,omitempty"`
}

func (x *Classifier) Reset() {
	*x = Classifier{}
	if protoimpl.UnsafeEnabled {
		mi := &file_pb_classifier_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Classifier) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Classifier) ProtoMessage() {}

func (x *Classifier) ProtoReflect() protoreflect.Message {
	mi := &file_pb_classifier_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Classifier.ProtoReflect.Descriptor instead.
func (*Classifier) Descriptor() ([]byte, []int) {
	return file_pb_classifier_proto_rawDescGZIP(), []int{0}
}

func (x *Classifier) GetLearned() float64 {
	if x != nil {
		return x.Learned
	}
	return 0
}

func (x *Classifier) GetMinProb() float64 {
	if x != nil {
		return x.MinProb
	}
	return 0
}

func (x *Classifier) GetClassInt32() []int32 {
	if x != nil {
		return x.ClassInt32
	}
	return nil
}

func (x *Classifier) GetClassInt64() []int64 {
	if x != nil {
		return x.ClassInt64
	}
	return nil
}

func (x *Classifier) GetClassUint32() []uint32 {
	if x != nil {
		return x.ClassUint32
	}
	return nil
}

func (x *Classifier) GetClassUint64() []uint64 {
	if x != nil {
		return x.ClassUint64
	}
	return nil
}

func (x *Classifier) GetClassFloat() []float32 {
	if x != nil {
		return x.ClassFloat
	}
	return nil
}

func (x *Classifier) GetClassDouble() []float64 {
	if x != nil {
		return x.ClassDouble
	}
	return nil
}

func (x *Classifier) GetClassString() []string {
	if x != nil {
		return x.ClassString
	}
	return nil
}

func (x *Classifier) GetPrior() []float64 {
	if x != nil {
		return x.Prior
	}
	return nil
}

func (x *Classifier) GetProb() []*Prob {
	if x != nil {
		return x.Prob
	}
	return nil
}

func (x *Classifier) GetValueInt32() []int32 {
	if x != nil {
		return x.ValueInt32
	}
	return nil
}

func (x *Classifier) GetValueInt64() []int64 {
	if x != nil {
		return x.ValueInt64
	}
	return nil
}

func (x *Classifier) GetValueUint32() []uint32 {
	if x != nil {
		return x.ValueUint32
	}
	return nil
}

func (x *Classifier) GetValueUint64() []uint64 {
	if x != nil {
		return x.ValueUint64
	}
	return nil
}

func (x *Classifier) GetValueFloat() []float32 {
	if x != nil {
		return x.ValueFloat
	}
	return nil
}

func (x *Classifier) GetValueDouble() []float64 {
	if x != nil {
		return x.ValueDouble
	}
	return nil
}

func (x *Classifier) GetValueString() []string {
	if x != nil {
		return x.ValueString
	}
	return nil
}

var File_pb_classifier_proto protoreflect.FileDescriptor

var file_pb_classifier_proto_rawDesc = []byte{
	0x0a, 0x13, 0x70, 0x62, 0x2f, 0x63, 0x6c, 0x61, 0x73, 0x73, 0x69, 0x66, 0x69, 0x65, 0x72, 0x2e,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x02, 0x70, 0x62, 0x1a, 0x0d, 0x70, 0x62, 0x2f, 0x70, 0x72,
	0x6f, 0x62, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22, 0xc4, 0x04, 0x0a, 0x0a, 0x43, 0x6c, 0x61,
	0x73, 0x73, 0x69, 0x66, 0x69, 0x65, 0x72, 0x12, 0x18, 0x0a, 0x07, 0x6c, 0x65, 0x61, 0x72, 0x6e,
	0x65, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x01, 0x52, 0x07, 0x6c, 0x65, 0x61, 0x72, 0x6e, 0x65,
	0x64, 0x12, 0x18, 0x0a, 0x07, 0x6d, 0x69, 0x6e, 0x50, 0x72, 0x6f, 0x62, 0x18, 0x02, 0x20, 0x01,
	0x28, 0x01, 0x52, 0x07, 0x6d, 0x69, 0x6e, 0x50, 0x72, 0x6f, 0x62, 0x12, 0x1e, 0x0a, 0x0a, 0x43,
	0x6c, 0x61, 0x73, 0x73, 0x49, 0x6e, 0x74, 0x33, 0x32, 0x18, 0x04, 0x20, 0x03, 0x28, 0x05, 0x52,
	0x0a, 0x43, 0x6c, 0x61, 0x73, 0x73, 0x49, 0x6e, 0x74, 0x33, 0x32, 0x12, 0x1e, 0x0a, 0x0a, 0x43,
	0x6c, 0x61, 0x73, 0x73, 0x49, 0x6e, 0x74, 0x36, 0x34, 0x18, 0x05, 0x20, 0x03, 0x28, 0x03, 0x52,
	0x0a, 0x43, 0x6c, 0x61, 0x73, 0x73, 0x49, 0x6e, 0x74, 0x36, 0x34, 0x12, 0x20, 0x0a, 0x0b, 0x43,
	0x6c, 0x61, 0x73, 0x73, 0x55, 0x69, 0x6e, 0x74, 0x33, 0x32, 0x18, 0x06, 0x20, 0x03, 0x28, 0x0d,
	0x52, 0x0b, 0x43, 0x6c, 0x61, 0x73, 0x73, 0x55, 0x69, 0x6e, 0x74, 0x33, 0x32, 0x12, 0x20, 0x0a,
	0x0b, 0x43, 0x6c, 0x61, 0x73, 0x73, 0x55, 0x69, 0x6e, 0x74, 0x36, 0x34, 0x18, 0x07, 0x20, 0x03,
	0x28, 0x04, 0x52, 0x0b, 0x43, 0x6c, 0x61, 0x73, 0x73, 0x55, 0x69, 0x6e, 0x74, 0x36, 0x34, 0x12,
	0x1e, 0x0a, 0x0a, 0x43, 0x6c, 0x61, 0x73, 0x73, 0x46, 0x6c, 0x6f, 0x61, 0x74, 0x18, 0x08, 0x20,
	0x03, 0x28, 0x02, 0x52, 0x0a, 0x43, 0x6c, 0x61, 0x73, 0x73, 0x46, 0x6c, 0x6f, 0x61, 0x74, 0x12,
	0x20, 0x0a, 0x0b, 0x43, 0x6c, 0x61, 0x73, 0x73, 0x44, 0x6f, 0x75, 0x62, 0x6c, 0x65, 0x18, 0x09,
	0x20, 0x03, 0x28, 0x01, 0x52, 0x0b, 0x43, 0x6c, 0x61, 0x73, 0x73, 0x44, 0x6f, 0x75, 0x62, 0x6c,
	0x65, 0x12, 0x20, 0x0a, 0x0b, 0x43, 0x6c, 0x61, 0x73, 0x73, 0x53, 0x74, 0x72, 0x69, 0x6e, 0x67,
	0x18, 0x0a, 0x20, 0x03, 0x28, 0x09, 0x52, 0x0b, 0x43, 0x6c, 0x61, 0x73, 0x73, 0x53, 0x74, 0x72,
	0x69, 0x6e, 0x67, 0x12, 0x14, 0x0a, 0x05, 0x70, 0x72, 0x69, 0x6f, 0x72, 0x18, 0x0b, 0x20, 0x03,
	0x28, 0x01, 0x52, 0x05, 0x70, 0x72, 0x69, 0x6f, 0x72, 0x12, 0x1c, 0x0a, 0x04, 0x70, 0x72, 0x6f,
	0x62, 0x18, 0x0c, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x08, 0x2e, 0x70, 0x62, 0x2e, 0x50, 0x72, 0x6f,
	0x62, 0x52, 0x04, 0x70, 0x72, 0x6f, 0x62, 0x12, 0x1e, 0x0a, 0x0a, 0x56, 0x61, 0x6c, 0x75, 0x65,
	0x49, 0x6e, 0x74, 0x33, 0x32, 0x18, 0x0d, 0x20, 0x03, 0x28, 0x05, 0x52, 0x0a, 0x56, 0x61, 0x6c,
	0x75, 0x65, 0x49, 0x6e, 0x74, 0x33, 0x32, 0x12, 0x1e, 0x0a, 0x0a, 0x56, 0x61, 0x6c, 0x75, 0x65,
	0x49, 0x6e, 0x74, 0x36, 0x34, 0x18, 0x0e, 0x20, 0x03, 0x28, 0x03, 0x52, 0x0a, 0x56, 0x61, 0x6c,
	0x75, 0x65, 0x49, 0x6e, 0x74, 0x36, 0x34, 0x12, 0x20, 0x0a, 0x0b, 0x56, 0x61, 0x6c, 0x75, 0x65,
	0x55, 0x69, 0x6e, 0x74, 0x33, 0x32, 0x18, 0x0f, 0x20, 0x03, 0x28, 0x0d, 0x52, 0x0b, 0x56, 0x61,
	0x6c, 0x75, 0x65, 0x55, 0x69, 0x6e, 0x74, 0x33, 0x32, 0x12, 0x20, 0x0a, 0x0b, 0x56, 0x61, 0x6c,
	0x75, 0x65, 0x55, 0x69, 0x6e, 0x74, 0x36, 0x34, 0x18, 0x10, 0x20, 0x03, 0x28, 0x04, 0x52, 0x0b,
	0x56, 0x61, 0x6c, 0x75, 0x65, 0x55, 0x69, 0x6e, 0x74, 0x36, 0x34, 0x12, 0x1e, 0x0a, 0x0a, 0x56,
	0x61, 0x6c, 0x75, 0x65, 0x46, 0x6c, 0x6f, 0x61, 0x74, 0x18, 0x11, 0x20, 0x03, 0x28, 0x02, 0x52,
	0x0a, 0x56, 0x61, 0x6c, 0x75, 0x65, 0x46, 0x6c, 0x6f, 0x61, 0x74, 0x12, 0x20, 0x0a, 0x0b, 0x56,
	0x61, 0x6c, 0x75, 0x65, 0x44, 0x6f, 0x75, 0x62, 0x6c, 0x65, 0x18, 0x12, 0x20, 0x03, 0x28, 0x01,
	0x52, 0x0b, 0x56, 0x61, 0x6c, 0x75, 0x65, 0x44, 0x6f, 0x75, 0x62, 0x6c, 0x65, 0x12, 0x20, 0x0a,
	0x0b, 0x56, 0x61, 0x6c, 0x75, 0x65, 0x53, 0x74, 0x72, 0x69, 0x6e, 0x67, 0x18, 0x13, 0x20, 0x03,
	0x28, 0x09, 0x52, 0x0b, 0x56, 0x61, 0x6c, 0x75, 0x65, 0x53, 0x74, 0x72, 0x69, 0x6e, 0x67, 0x42,
	0x06, 0x5a, 0x04, 0x2e, 0x2f, 0x70, 0x62, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_pb_classifier_proto_rawDescOnce sync.Once
	file_pb_classifier_proto_rawDescData = file_pb_classifier_proto_rawDesc
)

func file_pb_classifier_proto_rawDescGZIP() []byte {
	file_pb_classifier_proto_rawDescOnce.Do(func() {
		file_pb_classifier_proto_rawDescData = protoimpl.X.CompressGZIP(file_pb_classifier_proto_rawDescData)
	})
	return file_pb_classifier_proto_rawDescData
}

var file_pb_classifier_proto_msgTypes = make([]protoimpl.MessageInfo, 1)
var file_pb_classifier_proto_goTypes = []interface{}{
	(*Classifier)(nil), // 0: pb.Classifier
	(*Prob)(nil),       // 1: pb.Prob
}
var file_pb_classifier_proto_depIdxs = []int32{
	1, // 0: pb.Classifier.prob:type_name -> pb.Prob
	1, // [1:1] is the sub-list for method output_type
	1, // [1:1] is the sub-list for method input_type
	1, // [1:1] is the sub-list for extension type_name
	1, // [1:1] is the sub-list for extension extendee
	0, // [0:1] is the sub-list for field type_name
}

func init() { file_pb_classifier_proto_init() }
func file_pb_classifier_proto_init() {
	if File_pb_classifier_proto != nil {
		return
	}
	file_pb_prob_proto_init()
	if !protoimpl.UnsafeEnabled {
		file_pb_classifier_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Classifier); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_pb_classifier_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   1,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_pb_classifier_proto_goTypes,
		DependencyIndexes: file_pb_classifier_proto_depIdxs,
		MessageInfos:      file_pb_classifier_proto_msgTypes,
	}.Build()
	File_pb_classifier_proto = out.File
	file_pb_classifier_proto_rawDesc = nil
	file_pb_classifier_proto_goTypes = nil
	file_pb_classifier_proto_depIdxs = nil
}
