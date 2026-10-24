"""Three-class YOLO11 ONNX CPU protocol; no BPU execution is claimed here."""
import cv2
import numpy as np

NAMES=('panel','needle','center_region')


def preprocess(image):
    h,w=image.shape[:2]
    scale=min(640/w,640/h)
    nw,nh=round(w*scale),round(h*scale)
    left,top=round((640-nw)/2-.1),round((640-nh)/2-.1)
    resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    padded=cv2.copyMakeBorder(resized,top,640-nh-top,left,640-nw-left,
                             cv2.BORDER_CONSTANT,value=(114,114,114))
    tensor=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
    # Rounding makes horizontal/vertical gains slightly different. Restore
    # coordinates using the actual integer resize, matching Ultralytics.
    return tensor,(float(nw/w),float(nh/h),float(left),float(top))


def iou(box,boxes):
    size=np.maximum(0,np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2]))
    area=size[:,0]*size[:,1]
    a=max(0,box[2]-box[0])*max(0,box[3]-box[1])
    b=np.maximum(0,boxes[:,2]-boxes[:,0])*np.maximum(0,boxes[:,3]-boxes[:,1])
    return area/np.maximum(a+b-area,1e-9)


def decode(output,original_shape,mapping,confidence=.25,iou_threshold=.45,max_det=300):
    output=np.asarray(output)
    if output.shape!=(1,7,8400) or not np.isfinite(output).all():
        raise ValueError(f'Expected finite packed [1,7,8400] output, got {output.shape}.')
    rows=output[0].T
    classes=rows[:,4:].argmax(axis=1)
    scores=rows[np.arange(len(rows)),4+classes]
    selected=scores>confidence
    rows,classes,scores=rows[selected],classes[selected],scores[selected]
    if not len(rows):
        return []
    boxes=np.column_stack((rows[:,:2]-rows[:,2:4]/2,rows[:,:2]+rows[:,2:4]/2))
    if len(mapping)==3:
        gainx,padx,pady=mapping;gainy=gainx
    else:
        gainx,gainy,padx,pady=mapping
    boxes[:,[0,2]]=(boxes[:,[0,2]]-padx)/gainx
    boxes[:,[1,3]]=(boxes[:,[1,3]]-pady)/gainy
    h,w=original_shape[:2]
    boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,w)
    boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,h)
    order=np.argsort(-scores,kind='stable')
    keep=[]
    while len(order) and len(keep)<max_det:
        index=int(order[0]);keep.append(index)
        rest=order[1:]
        order=rest[(classes[rest]!=classes[index]) | (iou(boxes[index],boxes[rest])<=iou_threshold)]
    return [{'label':NAMES[int(classes[i])],'class_id':int(classes[i]),
             'score':float(scores[i]),'box':boxes[i].astype(float).tolist()} for i in keep]


def raw_heads_to_packed(outputs):
    """Decode the pinned X5 export's six NHWC floating heads for CPU comparison.

    Compiled BIN output is a separate quantized runtime protocol and must use
    the matching official decoder; do not feed quantized tensors here.
    """
    if len(outputs)!=6:
        raise ValueError('Expected [class,box] for strides 8,16,32.')
    groups=[]
    for index,(size,stride) in enumerate([(80,8),(40,16),(20,32)]):
        cls,box=[np.asarray(v,dtype=np.float32) for v in outputs[index*2:index*2+2]]
        if cls.shape!=(1,size,size,3) or box.shape!=(1,size,size,64):
            raise ValueError(f'Unexpected raw head shapes {cls.shape}, {box.shape}.')
        if not np.isfinite(cls).all() or not np.isfinite(box).all():
            raise ValueError('Nonfinite raw heads.')
        logits=box.reshape(-1,4,16)
        exp=np.exp(logits-logits.max(axis=2,keepdims=True))
        distance=((exp/exp.sum(axis=2,keepdims=True))*np.arange(16,dtype=np.float32)).sum(axis=2)
        yy,xx=np.indices((size,size))
        center=np.column_stack([xx.ravel()+.5,yy.ravel()+.5])
        a,b=(center-distance[:,:2])*stride,(center+distance[:,2:])*stride
        scores=1/(1+np.exp(-np.clip(cls.reshape(-1,3),-80,80)))
        groups.append(np.column_stack([(a+b)/2,b-a,scores]))
    return np.ascontiguousarray(np.concatenate(groups).T[None],dtype=np.float32)
