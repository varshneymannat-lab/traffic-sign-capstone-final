from scripts.review_mtsd_annotations import compare_labels


def fixture():
    annotation={'width':100,'height':100,'objects':[
        {'label':'stop','bbox':{'xmin':10,'ymin':20,'xmax':30,'ymax':40}},
        {'label':'other','bbox':{'xmin':50,'ymin':50,'xmax':60,'ymax':60}}]}
    row={'sign_label':'stop','x_center':.2,'y_center':.3,'box_width':.2,'box_height':.2}
    return annotation,row


def test_keeps_in_scope_and_allows_out_of_scope_background():
    annotation,row=fixture()
    assert compare_labels(annotation,[row],{'stop':'stop'})


def test_missing_target_is_detected():
    annotation,_=fixture()
    assert not compare_labels(annotation,[],{'stop':'stop'})


def test_wrong_box_is_detected():
    annotation,row=fixture()
    row['x_center']=.4
    assert not compare_labels(annotation,[row],{'stop':'stop'})
